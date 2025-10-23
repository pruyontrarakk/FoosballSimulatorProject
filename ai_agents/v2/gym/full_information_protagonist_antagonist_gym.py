import math
import os
from pathlib import Path
from typing import Optional, Tuple, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

# from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin  # (unused, safe to remove)

DIRECTION_CHANGE = 1
TABLE_MAX_Y_DIM = 65
BALL_STOPPED_COUNT_THRESHOLD = 10

# Default sim XML path (can be overridden by passing xml_path arg)
SIM_PATH = os.environ.get(
    "SIM_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "../../../..",
        "foosball_sim",
        "v2",
        "foosball_sim.xml",
    ),
)
SIM_PATH = os.path.abspath(SIM_PATH)

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]


class FoosballEnv(gym.Env):
    """
    Foosball MuJoCo environment (v2) with offscreen rendering support.

    Set render_mode="rgb_array" to enable frame rendering compatible with
    Gymnasium's RecordVideo wrapper.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        antagonist_model=None,
        xml_path: Optional[str] = None,
        play_until_goal: bool = False,
        verbose_mode: bool = False,
        max_steps: int = 1000,
        seed: int = 0,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        self.render_mode = render_mode
        self.play_until_goal = play_until_goal
        self.verbose_mode = verbose_mode
        self.max_steps = max_steps

        # Resolve XML path
        if xml_path is None:
            repo_root = Path(__file__).resolve().parents[3]  # .../FoosballSimulatorProject
            xml_path = repo_root / "foosball_sim" / "v2" / "foosball_sim.xml"

        xml_path = Path(xml_path).resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        # Ensure relative assets referenced in the XML are found
        os.chdir(xml_path.parent)

        # --- Create model/data FIRST ---
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # --- Offscreen renderer (created AFTER model/data exists) ---
        self._renderer = None
        self._last_frame = None
        if self.render_mode == "rgb_array":
            # Use XML <visual><global offwidth/height> if provided, else fall back.
            try:
                offw = int(getattr(self.model.vis.global_, "offwidth", 0)) or 640
                offh = int(getattr(self.model.vis.global_, "offheight", 0)) or 480
            except Exception:
                offw, offh = 640, 480
            self._renderer = mujoco.Renderer(self.model, width=offw, height=offh)

        # ---- Env-specific setup ----
        self.simulation_time = 0.0

        self.num_rods_per_player = 4
        self.num_players = 2
        self.num_rods = self.num_rods_per_player * self.num_players

        self.protagonist_action_size = self.num_rods_per_player * 2  # 8 actions
        self.antagonist_action_size = self.num_rods_per_player * 2   # 8 actions

        action_high = np.ones(self.protagonist_action_size, dtype=np.float32)

        # TEMP: combined action box used by agents
        self.action_space = spaces.Box(
            low=-20.0 * action_high, high=20.0 * action_high, dtype=np.float32
        )

        # Observation: ball(3) + ball_vel(3) + each player's rods:
        #   4 slide pos + 4 slide vel + 4 rotate pos + 4 rotate vel = 16 per player
        #   two players => 32; total 3+3+32 = 38
        obs_dim = 38
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Rewards / termination helpers
        self._healthy_reward = 1.0
        self._ctrl_cost_weight = 0.005
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (-80.0, 80.0)
        self.max_no_progress_steps = 15

        self.prev_ball_y: Optional[float] = None
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

        self.antagonist_model = antagonist_model

        # Optional seeding
        if seed is not None:
            np.random.seed(seed)

    # ------------- Gymnasium API -------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Place ball with small random perturbation
        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")

        x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
        y_qpos_adr = self.model.jnt_qposadr[ball_y_id]

        xy_random = np.random.normal(loc=[-0.5, 0.0], scale=[0.5, 0.5])
        self.data.qpos[x_qpos_adr] = xy_random[0]
        self.data.qpos[y_qpos_adr] = xy_random[1]

        self.simulation_time = 0.0
        self.prev_ball_y = float(self.data.qpos[y_qpos_adr])
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, protagonist_action: np.ndarray):
        protagonist_action = np.clip(
            protagonist_action, self.action_space.low, self.action_space.high
        )

        # Antagonist action (if model provided)
        antagonist_observation = self._get_antagonist_obs()
        if self.antagonist_model is not None:
            ant_action, *_ = self.antagonist_model.predict(antagonist_observation)
            ant_action = np.clip(ant_action, -1.0, 1.0)
            antagonist_action = self._adjust_antagonist_action(ant_action)
        else:
            antagonist_action = np.zeros(self.antagonist_action_size, dtype=np.float32)

        # Apply controls: protagonist first half, antagonist second half
        self.data.ctrl[: self.protagonist_action_size] = protagonist_action
        self.data.ctrl[
            self.protagonist_action_size : self.protagonist_action_size
            + self.antagonist_action_size
        ] = antagonist_action

        mujoco.mj_step(self.model, self.data)
        self.simulation_time += float(self.model.opt.timestep)

        obs = self._get_obs()
        reward = self.compute_reward(protagonist_action)
        terminated = self.terminated
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        """Return an RGB frame (H,W,3) when render_mode='rgb_array'."""
        if self.render_mode != "rgb_array":
            return None

        if self._renderer is None:
            # lazy-create if needed
            try:
                offw = int(getattr(self.model.vis.global_, "offwidth", 0)) or 640
                offh = int(getattr(self.model.vis.global_, "offheight", 0)) or 480
            except Exception:
                offw, offh = 640, 480
            self._renderer = mujoco.Renderer(self.model, width=offw, height=offh)

        self._renderer.update_scene(self.data)
        frame = self._renderer.render()
        self._last_frame = frame
        return frame

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    # ------------- Observations & helpers -------------

    def _get_ball_obs(self) -> Tuple[List[float], List[float]]:
        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
        ball_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_z")

        x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
        y_qpos_adr = self.model.jnt_qposadr[ball_y_id]
        z_qpos_adr = self.model.jnt_qposadr[ball_z_id]

        x_qvel_adr = self.model.jnt_dofadr[ball_x_id]
        y_qvel_adr = self.model.jnt_dofadr[ball_y_id]
        z_qvel_adr = self.model.jnt_dofadr[ball_z_id]

        ball_pos = [
            float(self.data.qpos[x_qpos_adr]),
            float(self.data.qpos[y_qpos_adr]),
            float(self.data.qpos[z_qpos_adr]),
        ]
        ball_vel = [
            float(self.data.qvel[x_qvel_adr]),
            float(self.data.qvel[y_qvel_adr]),
            float(self.data.qvel[z_qvel_adr]),
        ]
        return ball_pos, ball_vel

    def _get_antagonist_obs(self):
        # If/when you train an antagonist, return a proper observation here
        return None

    def _get_obs(self) -> np.ndarray:
        ball_pos, ball_vel = self._get_ball_obs()

        rod_slide_positions = []
        rod_slide_velocities = []
        rod_rotate_positions = []
        rod_rotate_velocities = []

        # Collect observations for both players' rods
        for player in ["y", "b"]:
            for rod in RODS:
                # Linear joints
                slide_joint_name = f"{player}{rod}linear"
                slide_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, slide_joint_name
                )
                slide_qpos_adr = self.model.jnt_qposadr[slide_joint_id]
                slide_qvel_adr = self.model.jnt_dofadr[slide_joint_id]
                rod_slide_positions.append(float(self.data.qpos[slide_qpos_adr]))
                rod_slide_velocities.append(float(self.data.qvel[slide_qvel_adr]))

                # Rotational joints
                rotate_joint_name = f"{player}{rod}rotation"
                rotate_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, rotate_joint_name
                )
                rotate_qpos_adr = self.model.jnt_qposadr[rotate_joint_id]
                rotate_qvel_adr = self.model.jnt_dofadr[rotate_joint_id]
                rod_rotate_positions.append(float(self.data.qpos[rotate_qpos_adr]))
                rod_rotate_velocities.append(float(self.data.qvel[rotate_qvel_adr]))

        obs = np.array(
            ball_pos
            + ball_vel
            + rod_slide_positions
            + rod_slide_velocities
            + rod_rotate_positions
            + rod_rotate_velocities,
            dtype=np.float32,
        )

        # Validate observation shape
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observation shape {obs.shape} != expected {self.observation_space.shape}"
            )
        return obs

    def _adjust_antagonist_action(self, antagonist_action: np.ndarray) -> np.ndarray:
        # simple policy: mirror protagonist directions
        adjusted_action = -np.asarray(antagonist_action, dtype=np.float32).copy()
        return adjusted_action

    # ------------- Rewards / termination -------------

    @staticmethod
    def euclidean_goal_distance(x: float, y: float) -> float:
        # Point (0, 64)
        return math.sqrt((x - 0.0) ** 2 + (y - TABLE_MAX_Y_DIM) ** 2)

    def compute_reward(self, protagonist_action: np.ndarray) -> float:
        (bx, by, _), _vel = self._get_ball_obs()
        inverse_distance_to_goal = 300.0 - self.euclidean_goal_distance(bx, by)
        if by > TABLE_MAX_Y_DIM:
            inverse_distance_to_goal = 0.0

        # 1-norm control penalty
        ctrl_cost = self._ctrl_cost_weight * float(np.sum(np.abs(protagonist_action))) * -1.0

        victory = 1000.0 * DIRECTION_CHANGE if by > TABLE_MAX_Y_DIM else 0.0
        loss = -1000.0 * DIRECTION_CHANGE if by < -TABLE_MAX_Y_DIM else 0.0

        reward = loss + victory + inverse_distance_to_goal + ctrl_cost
        return float(reward)

    @property
    def is_healthy(self) -> bool:
        ( _bx, _by, bz ), _ = self._get_ball_obs()
        min_z, max_z = self._healthy_z_range
        return bool(min_z < bz < max_z)

    def _is_ball_moving(self) -> bool:
        _pos, vel = self._get_ball_obs()
        return bool(np.linalg.norm(vel) > 0.5)

    def _determine_progression(self) -> None:
        (_bx, by, _bz), _ = self._get_ball_obs()
        if self.prev_ball_y is not None:
            if by > self.prev_ball_y:
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1
        self.prev_ball_y = by

    @property
    def terminated(self) -> bool:
        self._determine_progression()

        self.ball_stopped_count = 0 if self._is_ball_moving() else self.ball_stopped_count + 1
        ball_stagnant = self.ball_stopped_count >= BALL_STOPPED_COUNT_THRESHOLD

        over_max_steps = self.simulation_time >= float(self.max_steps)
        unhealthy = not self.is_healthy
        no_progress = self.no_progress_steps >= self.max_no_progress_steps

        (bx, by, _bz), _ = self._get_ball_obs()
        victory = (by < -TABLE_MAX_Y_DIM) or (by > TABLE_MAX_Y_DIM)

        if victory:
            print("Victory")
            print(f"Ball x: {bx}, Ball y: {by}")

        terminated = (
            unhealthy
            or (no_progress and not self.play_until_goal)
            or ball_stagnant
            or over_max_steps
            # or victory  # enable if you want episodes to end on any goal
        ) if self._terminate_when_unhealthy else False

        if self.verbose_mode and terminated:
            print("Terminated")
            print(
                f"Unhealthy: {unhealthy}, No progress: {no_progress}, "
                f"Victory: {victory}, Ball stagnant: {ball_stagnant}"
            )
            print(f"x: {bx}, y: {by}")
        return bool(terminated)
    