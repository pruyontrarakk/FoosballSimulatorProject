# ai_agents/common/train/impl/sac_agent.py
from __future__ import annotations
from typing import Optional, Any
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from ai_agents.common.train.interface.foosball_agent import FoosballAgent


class SACFoosballAgent(FoosballAgent):
    """
    Concrete implementation of FoosballAgent using Stable-Baselines3 SAC.
    Implements ALL abstract methods with compatible signatures.
    """

    def __init__(
        self,
        id: int,
        env: Any = None,
        log_dir: str = "./logs",
        model_dir: str = "./models",
        policy_kwargs: dict = dict(net_arch=[512, 512]),  
        device: str = "cuda",
        buffer_size: int = 300_000,
        batch_size: int = 128,
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
    ) -> None:
        self._id = id
        self.env = env
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.id_subdir = f"{model_dir}/{id}"
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate

        self.model: Optional[SAC] = None

        os.makedirs(self.id_subdir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # ---------- abstract interface: MUST match names/signatures ----------

    def get_id(self) -> int:
        return self._id

    def initialize_agent(self) -> None:
        """Create a fresh model or load if checkpoint exists."""
        try:
            self.load()  # will use default path + self.device
        except Exception:
            print(f"Agent {self._id} could not load model. Initializing new model.")
            self.model = SAC(
                "MlpPolicy",
                self.env,
                policy_kwargs=self.policy_kwargs,
                device=self.device,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                learning_rate=self.learning_rate,
                verbose=0,
            )
        print(f"Agent {self._id} initialized.")

    def predict(self, observation: Any, deterministic: bool = False) -> Any:
        """Return action from current policy."""
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def learn(self, total_timesteps: int) -> None:
        """Train for a number of timesteps."""
        if self.model is None:
            # create on demand if not initialized
            self.model = SAC(
                "MlpPolicy",
                self.env,
                policy_kwargs=self.policy_kwargs,
                device=self.device,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                learning_rate=self.learning_rate,
                verbose=0,
            )

        callback = self.create_callback(self.env)
        tb_log_name = f"sac_{self._id}"
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            progress_bar=True,
        )

    def save(self, path: Optional[str] = None) -> None:
        """Save model to path (defaults to best_model dir)."""
        if self.model is None:
            return
        if path is None:
            path = os.path.join(self.id_subdir, "sac", "best_model")
        os.makedirs(path, exist_ok=True)
        self.model.save(path)

    def load(self, path: Optional[str] = None) -> None:
        """Load model from path (defaults to best_model.zip)."""
        if path is None:
            path = os.path.join(self.id_subdir, "sac", "best_model", "best_model.zip")
        self.model = SAC.load(path, device=self.device)
        if self.env is not None:
            self.model.set_env(self.env)
        print(f"Agent {self._id} loaded model from {path}")

    def change_env(self, env: Any) -> None:
        """Swap environment used by this agent."""
        self.env = env
        if self.model is not None:
            self.model.set_env(env)

    # ---------- helpers ----------

    def create_callback(self, env: Any) -> EvalCallback:
        # keep eval light to save VRAM
        best_path = os.path.join(self.id_subdir, "sac", "best_model")
        os.makedirs(best_path, exist_ok=True)
        return EvalCallback(
            env,
            best_model_save_path=best_path,
            log_path=self.log_dir,
            eval_freq=10_000,      # less frequent eval
            n_eval_episodes=5,     # fewer episodes
            render=False,
            deterministic=True,
        )
