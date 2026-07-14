"""Capture a deterministic frame-zero observation from the MjLab environment."""

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
OUTPUT = Path(
    "/root/gpufree-data/RoboJuDo/backups/right_overhand_20260713_codex/"
    "mjlab_nominal_obs.npz"
)


def main() -> None:
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.events = {}
    motion_cfg = env_cfg.commands["motion"]
    assert isinstance(motion_cfg, MotionCommandCfg)
    motion_cfg.motion_file = str(
        RESULT_DIR / "motion/motion_home_adaptive_selected_v3_balanced.npz"
    )
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.sampling_mode = "start"

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

    observations = vec_env.get_observations()
    observation = observations["actor"][0].cpu().numpy().copy()
    with torch.no_grad():
        action = policy(observations)[0].cpu().numpy().copy()
    np.savez(OUTPUT, observation=observation, action=action)

    slices = {
        "command": slice(0, 58),
        "anchor_ori": slice(58, 64),
        "base_ang_vel": slice(64, 67),
        "joint_pos": slice(67, 96),
        "joint_vel": slice(96, 125),
        "last_action": slice(125, 154),
    }
    for name, segment in slices.items():
        values = observation[segment]
        print(
            name,
            f"norm={np.linalg.norm(values):.9f}",
            f"min={values.min():.9f}",
            f"max={values.max():.9f}",
        )
    print(f"action_norm={np.linalg.norm(action):.9f}")
    vec_env.close()


if __name__ == "__main__":
    main()
