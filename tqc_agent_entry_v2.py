import os
from pathlib import Path
import argparse
import torch

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from ai_agents.common.train.impl.generic_agent_manager import GenericAgentManager
from ai_agents.common.train.impl.single_player_training_engine import SinglePlayerTrainingEngine
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from ai_agents.common.train.impl.tqc_agent import TQCFoosballAgent

torch.backends.cuda.matmul.fp32_precision = "high"  # or "ieee"
torch.backends.cudnn.conv.fp32_precision = "tf32"

REPO_ROOT = Path(__file__).resolve().parent
XML_PATH = REPO_ROOT / "foosball_sim" / "v2" / "foosball_sim.xml"
assert XML_PATH.exists(), f"Missing XML at {XML_PATH}"

N_ENVS = int(os.getenv("N_ENVS", "8"))

def _make_single_env(seed: int = 0):
    def _init():
        env = FoosballEnv(antagonist_model=None, xml_path=str(XML_PATH), render_mode=None)
        env.reset(seed=seed)
        return env
    return _init

def make_train_env():
    env = SubprocVecEnv([_make_single_env(i) for i in range(N_ENVS)])
    env = VecMonitor(env)
    return env

def make_eval_env():
    return _make_single_env(123)()

def tqc_foosball_env_factory(*_, **__):
    return make_train_env()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test TQC model.")
    parser.add_argument("-t", "--test", action="store_true", help="Run a short test after training")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epoch-steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    args = parser.parse_args()

    N_ENVS = args.n_envs
    os.environ.setdefault("MUJOCO_GL", "egl")

    agent_manager = GenericAgentManager(1, tqc_foosball_env_factory, TQCFoosballAgent)
    agent_manager.initialize_training_agents()
    agent_manager.initialize_frozen_best_models()

    engine = SinglePlayerTrainingEngine(
        agent_manager=agent_manager,
        environment_generator=tqc_foosball_env_factory,
    )
    engine.train(total_epochs=args.epochs, epoch_timesteps=args.epoch_steps, cycle_timesteps=10_000)

    if args.test:
        from sb3_contrib import TQC
        from stable_baselines3.common.evaluation import evaluate_policy

        eval_env = make_eval_env()

        ckpt = REPO_ROOT / "models" / "0" / "tqc" / "best_model" / "best_model.zip"
        assert ckpt.exists(), f"Checkpoint not found: {ckpt}"

        eval_model = TQC.load(
            str(ckpt),
            env=eval_env,  
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        mean_r, std_r = evaluate_policy(eval_model, eval_env, n_eval_episodes=5, deterministic=True)
        print(f"[EVAL/TQC] mean_reward={mean_r:.2f} +/- {std_r:.2f}")
