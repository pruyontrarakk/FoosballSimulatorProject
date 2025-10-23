# sac_agent_entry_v2.py
import os
from pathlib import Path
import argparse
import torch

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from ai_agents.common.train.impl.generic_agent_manager import GenericAgentManager
from ai_agents.common.train.impl.sac_agent import SACFoosballAgent
from ai_agents.common.train.impl.single_player_training_engine import SinglePlayerTrainingEngine
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

# ---------- fast matmul on Ampere ----------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent
XML_PATH = REPO_ROOT / "foosball_sim" / "v2" / "foosball_sim.xml"
assert XML_PATH.exists(), f"Missing XML at {XML_PATH}"

# how many parallel envs (can also set N_ENVS env var)
N_ENVS = int(os.getenv("N_ENVS", "8"))

def _make_single_env(seed: int = 0):
    def _init():
        env = FoosballEnv(antagonist_model=None, xml_path=str(XML_PATH), render_mode=None)
        env.reset(seed=seed)
        return env
    return _init    

def make_train_env():
    """Vectorized env for fast training (CPU parallel)."""
    env = SubprocVecEnv([_make_single_env(i) for i in range(N_ENVS)])
    env = VecMonitor(env)
    return env

def make_eval_env():
    """Single env for evaluation / rollout."""
    return _make_single_env(123)()

# Backwards-compatible factory signature (engine may call with args)
def sac_foosball_env_factory(*_, **__):
    # Use vectorized env for training by default
    return make_train_env()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("-t", "--test", action="store_true", help="Run a short test after training")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epoch-steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    args = parser.parse_args()

    # allow overriding N_ENVS via CLI
    N_ENVS = args.n_envs

    os.environ.setdefault("MUJOCO_GL", "egl")

    # --- build manager/engine with vectorized training env ---
    agent_manager = GenericAgentManager(1, sac_foosball_env_factory, SACFoosballAgent)
    agent_manager.initialize_training_agents()

    # Initialize a frozen model too (kept minimal to save VRAM)
    agent_manager.initialize_frozen_best_models()

    # tune SAC defaults inside the agent via kwargs (smaller net, larger batch)
    # open ai_agents/common/train/impl/sac_agent.py and ensure defaults roughly:
    #   policy_kwargs=dict(net_arch=[256, 256])
    #   buffer_size=500_000, batch_size=1024, gradient_steps=1, train_freq=(1, "env_step")
    # You already have a convenient constructor; we'll reuse it as-is.

    engine = SinglePlayerTrainingEngine(
        agent_manager=agent_manager,
        environment_generator=sac_foosball_env_factory,
    )

    # --- train ---
    engine.train(total_epochs=args.epochs, epoch_timesteps=args.epoch_steps, cycle_timesteps=10_000)

    # --- eval/test on a single env (avoid VecEnv here) ---
    # --- eval/test on a single env (avoid VecEnv here) ---
    if args.test:
        from stable_baselines3 import SAC
        from stable_baselines3.common.evaluation import evaluate_policy

        # 1) build a single, non-vec eval env
        eval_env = make_eval_env()

        # 2) point to your best checkpoint
        ckpt = REPO_ROOT / "foosball_sim" / "v2" / "models" / "0" / "sac" / "best_model" / "best_model.zip"
        assert ckpt.exists(), f"Checkpoint not found: {ckpt}"

        # 3) load a *new* model for eval with this single env
        eval_model = SAC.load(str(ckpt), env=eval_env, device="cuda")

        # 4) quick quantitative eval
        mean_r, std_r = evaluate_policy(eval_model, eval_env, n_eval_episodes=5, deterministic=True)
        print(f"[EVAL] mean_reward={mean_r:.2f} +/- {std_r:.2f}")

        # 5) optional: manual rollout (uncomment to watch logs/prints)
        """
        obs, _ = eval_env.reset()
        done, truncated = False, False
        while not (done or truncated):
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
        """

