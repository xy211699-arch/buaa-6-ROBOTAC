"""Run isolated RoboJuDo action deployment diagnostics."""

import argparse

import mujoco
import numpy as np

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.utils.util_func import get_gravity_orientation


TRAINING_TORQUE_LIMITS = [
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-torque", action="store_true")
    parser.add_argument("--training-step", action="store_true")
    parser.add_argument("--reference-init", action="store_true")
    return parser.parse_args()


def tilt_angle(quat_xyzw):
    gravity = get_gravity_orientation(quat_xyzw)
    return np.arccos(np.clip(-gravity[2], -1.0, 1.0))


def main() -> None:
    args = parse_args()
    cfg = ConfigManager(config_name="g1_mjlab_loco_right_overhand").get_cfg()
    cfg.ctrl = []
    cfg.do_safety_check = False
    cfg.env.visualize_extras = False

    if args.training_torque:
        cfg.mimic_policies[0].action_dof.torque_limits = TRAINING_TORQUE_LIMITS
    if args.training_step:
        cfg.env.sim_dt = 0.005
        cfg.env.sim_decimation = 4

    pipeline_class = getattr(robojudo.pipeline, cfg.pipeline_type)
    pipeline = pipeline_class(cfg=cfg)
    for _ in range(100):
        pipeline.step()

    pipeline._switch_to_action()
    action_policy = pipeline.policy
    if args.reference_init:
        motion = np.load(action_policy.cfg_policy.motion_file)
        pipeline.env.data.qpos[:3] = motion["body_pos_w"][0, 0]
        pipeline.env.data.qpos[3:7] = motion["body_quat_w"][0, 0]
        pipeline.env.data.qpos[7:] = motion["joint_pos"][0]
        pipeline.env.data.qvel[:] = 0.0
        pipeline.env.data.ctrl[:] = 0.0
        mujoco.mj_forward(pipeline.env.model, pipeline.env.data)
        pipeline.env.update()
        action_policy.reset()

    print(
        "scenario",
        f"training_torque={args.training_torque}",
        f"training_step={args.training_step}",
        f"reference_init={args.reference_init}",
    )
    for step in range(144):
        pipeline.step()
        if step % 10 == 0 or step >= 135:
            print(
                f"step={step:03d}",
                f"frame={action_policy.motion_frame:03d}",
                f"height={pipeline.env.base_pos[2]:.6f}",
                f"tilt={tilt_angle(pipeline.env.base_quat):.6f}",
                f"action_norm={np.linalg.norm(action_policy.last_action):.6f}",
            )

    pipeline.env.shutdown()


if __name__ == "__main__":
    main()
