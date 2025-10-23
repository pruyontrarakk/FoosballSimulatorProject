from __future__ import annotations
from typing import Optional, Any
import os
import inspect

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
from ai_agents.common.train.interface.foosball_agent import FoosballAgent


def _filter_supported_kwargs(cls, **maybe_kwargs):
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in maybe_kwargs.items() if k in sig.parameters}


class TQCFoosballAgent(FoosballAgent):
    def __init__(
        self,
        id: int,
        env: Any = None,
        log_dir: str = "./logs",
        model_dir: str = "./models",
        policy_kwargs: dict = dict(net_arch=[512, 512]),
        device: str = "cuda",
        buffer_size: int = 500_000,
        batch_size: int = 1024,
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
        # TQC params (will be auto-dropped if not supported by your sb3-contrib)
        n_critics: int = 5,
        n_quantiles: int = 25,
        top_k: int = 20,
        train_freq: tuple = (1, "step"),    # <- SB3 expects "step" or "episode" (not "env_step")
        gradient_steps: int = 1,
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

        # extras filtered at call time against your installed TQC signature
        self.tqc_extra = dict(
            n_critics=n_critics,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop_per_net=max(n_quantiles - top_k, 0),
            train_freq=(train_freq[0], "step") if isinstance(train_freq, tuple) else (1, "step"),
            gradient_steps=gradient_steps,
        )

        self.model: Optional[TQC] = None
        os.makedirs(self.id_subdir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _make_model(self):
        base = dict(
            policy="MlpPolicy",
            env=self.env,
            policy_kwargs=self.policy_kwargs,
            device=self.device,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tau=self.tau,
            learning_rate=self.learning_rate,
            verbose=0,
        )
        extras = _filter_supported_kwargs(TQC, **self.tqc_extra)
        return TQC(**base, **extras)

    def get_id(self) -> int:
        return self._id

    def initialize_agent(self) -> None:
        try:
            self.load()
        except Exception:
            print(f"Agent {self._id} could not load model. Initializing new TQC model.")
            self.model = self._make_model()
        print(f"TQC Agent {self._id} initialized.")

    def predict(self, observation: Any, deterministic: bool = False) -> Any:
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def learn(self, total_timesteps: int) -> None:
        if self.model is None:
            self.model = self._make_model()
        callback = self.create_callback(self.env)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"tqc_{self._id}",
            progress_bar=True,
        )

    def save(self, path: Optional[str] = None) -> None:
        if self.model is None:
            return
        if path is None:
            # leave off ".zip"; SB3 appends it
            path = os.path.join(self.id_subdir, "tqc", "best_model", "best_model")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: Optional[str] = None) -> None:
        # First-run safe: if no file, leave self.model=None and let initialize_agent() build fresh
        if path is None:
            path = os.path.join(self.id_subdir, "tqc", "best_model", "best_model.zip")
        try:
            self.model = TQC.load(path, device=self.device)
            if self.env is not None:
                self.model.set_env(self.env)
            print(f"TQC Agent {self._id} loaded model from {path}")
        except FileNotFoundError:
            self.model = None

    def change_env(self, env: Any) -> None:
        self.env = env
        if self.model is not None:
            self.model.set_env(env)

    def create_callback(self, env: Any) -> EvalCallback:
        best_path = os.path.join(self.id_subdir, "tqc", "best_model")
        os.makedirs(best_path, exist_ok=True)
        return EvalCallback(
            env,
            best_model_save_path=best_path,
            log_path=self.log_dir,
            eval_freq=10_000,
            n_eval_episodes=5,
            render=False,
            deterministic=True,
        )
