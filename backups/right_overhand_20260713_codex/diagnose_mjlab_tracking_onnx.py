"""Compare PT and ONNX tracking policies in the original MjLab environment."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg

import src.tasks  # noqa: F401


TASK_ID = "Unitree-G1-Tracking-No-State-Estimation"
CHECKPOINT = Path(
    "/root/gpufree-data/unitree_rl_mjlab/result/motions/right_overhand/model_15500.pt"
)
MOTION = Path(
    "/root/gpufree-data/unitree_rl_mjlab/result/motions/right_overhand/motion/"
    "motion_home_adaptive_selected_v3_balanced.npz"
)
ONNX = Path(
    "/root/gpufree-data/unitree_rl_mjlab/result/motions/right_overhand/policy.onnx"
)


def main() -> None:
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    motion_cfg = env_cfg.commands["motion"]
    assert isinstance(motion_cfg, MotionCommandCfg)
    motion_cfg.motion_file = str(MOTION)

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device="cpu")
    runner.load(
        str(CHECKPOINT),
        load_cfg={"actor": True},
        strict=True,
        map_location="cpu",
    )
    policy = runner.get_inference_policy(device="cpu")
    onnx_session = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    for step in range(150):
        with torch.no_grad():
            observations = vec_env.get_observations()
            actions = policy(observations)
        actor_obs = observations["actor"].cpu().numpy()
        onnx_actions = onnx_session.run(["actions"], {"obs": actor_obs})[0]
        action_diff = np.max(np.abs(actions.cpu().numpy() - onnx_actions))

        command = vec_env.unwrapped.command_manager.get_term("motion")
        _, _, dones, _ = vec_env.step(actions)
        if step % 10 == 0 or step >= 135 or bool(dones[0]):
            print(
                f"step={step:03d}",
                f"frame={int(command.time_steps[0]):03d}",
                f"pt_norm={np.linalg.norm(actions[0].cpu().numpy()):.6f}",
                f"onnx_norm={np.linalg.norm(onnx_actions[0]):.6f}",
                f"onnx_diff={action_diff:.8f}",
                f"done={int(dones[0])}",
            )

    vec_env.close()


if __name__ == "__main__":
    main()
