import os
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

REPO_ROOT = Path(__file__).resolve().parent
XML_PATH = REPO_ROOT / "foosball_sim" / "v2" / "foosball_sim.xml"

# Prefer ./models (your TQC path), fallback to foosball_sim tree if needed
MODELS_ROOT = REPO_ROOT / "models"
SAC_PATHS = [
    MODELS_ROOT / "0" / "sac" / "best_model" / "best_model.zip",
    REPO_ROOT / "foosball_sim" / "v2" / "models" / "0" / "sac" / "best_model" / "best_model.zip",
]
TQC_PATHS = [
    MODELS_ROOT / "0" / "tqc" / "best_model" / "best_model.zip",
    REPO_ROOT / "foosball_sim" / "v2" / "models" / "0" / "tqc" / "best_model" / "best_model.zip",
]

OUT_DIR = REPO_ROOT / "videos"
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_FILE = OUT_DIR / "rollout.mp4"

os.environ.setdefault("MUJOCO_GL", "egl")

torch.backends.cuda.matmul.fp32_precision = "high"
torch.backends.cudnn.conv.fp32_precision = "tf32"

env = FoosballEnv(
    antagonist_model=None,
    xml_path=str(XML_PATH),
    render_mode="rgb_array",
    play_until_goal=True,
    verbose_mode=False,
    max_steps=10_000,
)
env._terminate_when_unhealthy = False
env.max_no_progress_steps = 5_000

device = "cuda" if torch.cuda.is_available() else "cpu"

def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

policy = None
algo_name = "random"

tqc_ckpt = _first_existing(TQC_PATHS)
sac_ckpt = _first_existing(SAC_PATHS)

if tqc_ckpt is not None:
    from sb3_contrib import TQC
    policy = TQC.load(tqc_ckpt.as_posix(), env=env, device=device)
    algo_name = "TQC"
elif sac_ckpt is not None:
    from stable_baselines3 import SAC
    policy = SAC.load(sac_ckpt.as_posix(), env=env, device=device)
    algo_name = "SAC"

fps = 30
seconds = 20
total_steps = fps * seconds

writer = imageio.get_writer(
    OUT_FILE.as_posix(),
    format="FFMPEG",
    mode="I",
    fps=fps,
    codec="libx264",
    quality=8,
    macro_block_size=None,
    pixelformat="yuv420p",
)

obs, _ = env.reset(seed=42)

try:
    frame = env.render()
    if frame is not None:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame)

    for _ in range(total_steps):
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = np.random.uniform(low=-0.2, high=0.2, size=env.action_space.shape).astype("float32")

        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)

        if terminated or truncated:
            obs, _ = env.reset()

finally:
    writer.close()
    env.close()

print(f"[{algo_name}] Saved video to: {OUT_FILE}")
