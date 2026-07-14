"""Capture original MjLab actor observations and PT policy outputs."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg

import src.tasks  # noqa: F401


TASK_ID = "Unitree-G1-Tracking-No-State-Estimation"
RESULT_DIR = Path("/root/gpufree-data/unitree_rl_mjlab/result/motions/right_overhand")
TRACE_FILE = Path(
    "/root/gpufree-data/RoboJuDo/backups/right_overhand_20260713_codex/"
    "mjlab_pt_trace.npz"
)


def main() -> None:
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    motion_cfg = env_cfg.commands["motion"]
    assert isinstance(motion_cfg, MotionCommandCfg)
    motion_cfg.motion_file = str(
        RESULT_DIR / "motion/motion_home_adaptive_selected_v3_balanced.npz"
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device="cpu")
    runner.load(
        str(RESULT_DIR / "model_15500.pt"),
        load_cfg={"actor": True},
        strict=True,
        map_location="cpu",
    )
    policy = runner.get_inference_policy(device="cpu")

    observations_out = []
    actions_out = []
    for _ in range(150):
        with torch.no_grad():
            observations = vec_env.get_observations()
            actions = policy(observations)
        observations_out.append(observations["actor"][0].cpu().numpy().copy())
        actions_out.append(actions[0].cpu().numpy().copy())
        vec_env.step(actions)

    np.savez(
        TRACE_FILE,
        observations=np.asarray(observations_out, dtype=np.float32),
        actions=np.asarray(actions_out, dtype=np.float32),
    )
    print(f"saved={TRACE_FILE} samples={len(observations_out)}")
    vec_env.close()


if __name__ == "__main__":
    main()
