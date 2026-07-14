"""Compare RoboJuDo's frame-zero observation with deterministic MjLab output."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager


EXPECTED = (
    "/root/gpufree-data/RoboJuDo/backups/right_overhand_20260713_codex/"
    "mjlab_nominal_obs.npz"
)


def main() -> None:
    cfg = ConfigManager(config_name="g1_mjlab_loco_right_overhand").get_cfg()
    cfg.ctrl = []
    cfg.do_safety_check = False
    cfg.env.visualize_extras = False
    pipeline_class = getattr(robojudo.pipeline, cfg.pipeline_type)
    pipeline = pipeline_class(cfg=cfg)
    pipeline._switch_to_action()
    action_policy = pipeline.policy

    motion = np.load(action_policy.cfg_policy.motion_file)
    root_quat_wxyz = motion["body_quat_w"][0, 0]
    root_rotation = Rotation.from_quat(root_quat_wxyz[[1, 2, 3, 0]])
    pipeline.env.data.qpos[:3] = motion["body_pos_w"][0, 0]
    pipeline.env.data.qpos[3:7] = root_quat_wxyz
    pipeline.env.data.qpos[7:] = motion["joint_pos"][0]
    pipeline.env.data.qvel[:3] = motion["body_lin_vel_w"][0, 0]
    pipeline.env.data.qvel[3:6] = root_rotation.inv().apply(
        motion["body_ang_vel_w"][0, 0]
    )
    pipeline.env.data.qvel[6:] = motion["joint_vel"][0]
    pipeline.env.data.ctrl[:] = 0.0
    mujoco.mj_forward(pipeline.env.model, pipeline.env.data)
    pipeline.env.update()
    action_policy.reset()

    env_data = pipeline.env.get_data()
    ctrl_data = pipeline.ctrl_manager.get_ctrl_data(env_data)
    actual, _ = action_policy.get_observation(env_data, ctrl_data)
    actual_action = action_policy.get_action(actual)
    expected_data = np.load(EXPECTED)
    expected = expected_data["observation"]
    expected_action = expected_data["action"]

    slices = {
        "command": slice(0, 58),
        "anchor_ori": slice(58, 64),
        "base_ang_vel": slice(64, 67),
        "joint_pos": slice(67, 96),
        "joint_vel": slice(96, 125),
        "last_action": slice(125, 154),
    }
    for name, segment in slices.items():
        difference = np.abs(actual[segment] - expected[segment])
        print(
            name,
            f"max_abs_diff={difference.max():.9f}",
            f"mean_abs_diff={difference.mean():.9f}",
        )
    action_difference = np.abs(actual_action - expected_action)
    print(f"action_max_abs_diff={action_difference.max():.9f}")
    print(f"action_mean_abs_diff={action_difference.mean():.9f}")
    pipeline.env.shutdown()


if __name__ == "__main__":
    main()
