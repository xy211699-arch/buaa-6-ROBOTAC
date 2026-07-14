import tempfile
import unittest
from pathlib import Path

import numpy as np
from box import Box
from scipy.spatial.transform import Rotation as sRot

from robojudo.config.g1.policy.g1_mjlab_tracking_policy_cfg import (
    G1MjlabTrackingPolicyCfg,
)
from robojudo.policy.mjlab_tracking_policy import MjlabTrackingPolicy
from robojudo.config.g1.policy.g1_mjlab_velocity_policy_cfg import (
    G1MjlabVelocityPolicyCfg,
)
from robojudo.pipeline.mjlab_loco_action_pipeline import (
    ControlState,
    MjlabLocoActionPipeline,
)
from robojudo.pipeline.rl_loco_mimic_pipeline import PolicyInterpManager
from robojudo.policy.mjlab_velocity_policy import MjlabVelocityPolicy


LOCO_MODEL_PATH = (
    "/root/gpufree-data/RoboJuDo/assets/models/g1/mjlab/"
    "locomotion_v3/policy.onnx"
)
MODEL_PATH = (
    "/root/gpufree-data/unitree_rl_mjlab/result/motions/right_overhand/"
    "policy.onnx"
)


class TestMjlabTrackingPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.motion_path = Path(cls.temp_dir.name) / "motion.npz"

        joint_pos = np.stack(
            [np.arange(29, dtype=np.float32), np.arange(29, dtype=np.float32) + 30]
        )
        joint_vel = np.stack(
            [np.arange(29, dtype=np.float32) + 30, np.arange(29, dtype=np.float32) + 60]
        )
        body_quat_w = np.zeros((2, 30, 4), dtype=np.float32)
        body_quat_w[..., 0] = 1.0  # identity in the motion file's wxyz convention
        np.savez(
            cls.motion_path,
            fps=np.array([50.0]),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_quat_w=body_quat_w,
        )

        cls.policy = MjlabTrackingPolicy(
            cfg_policy=G1MjlabTrackingPolicyCfg(
                model_path=MODEL_PATH,
                motion_path=cls.motion_path.as_posix(),
            ),
            device="cpu",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        self.policy.reset()
        self.env_data = Box(
            {
                "base_ang_vel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "torso_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "dof_pos": self.policy.default_dof_pos
                + np.arange(29, dtype=np.float32)
                + 0.5,
                "dof_vel": np.arange(29, dtype=np.float32) + 60,
            }
        )

    def test_model_and_motion_contract(self):
        self.assertEqual(self.policy.anchor_body_index, 15)
        self.assertEqual(self.policy.observation_size, 154)
        self.assertEqual(self.policy.num_actions, 29)
        self.assertEqual(self.policy.motion_frame_count, 2)
        self.assertEqual(self.policy.motion_fps, 50.0)
        self.assertEqual(self.policy.action_scales.shape, (29,))

    def test_observation_order_is_exactly_154_dimensions(self):
        self.policy.last_action = np.arange(29, dtype=np.float32) + 60

        obs, extras = self.policy.get_observation(self.env_data, Box())

        expected_reference = np.concatenate(
            [
                np.arange(29, dtype=np.float32),
                np.arange(29, dtype=np.float32) + 30,
                np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            ]
        )
        expected_robot_state = np.concatenate(
            [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.arange(29, dtype=np.float32) + 0.5,
                np.arange(29, dtype=np.float32) + 60,
                np.arange(29, dtype=np.float32) + 60,
            ]
        )

        self.assertEqual(obs.shape, (154,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[:64], expected_reference)
        np.testing.assert_allclose(obs[64:], expected_robot_state)
        self.assertEqual(extras["motion_frame"], 0)
        self.assertEqual(extras["CALLBACK"], [])

    def test_reference_heading_alignment_uses_root_not_torso(self):
        original_quats = self.policy.reference_body_quat_wxyz.copy()
        try:
            self.policy.reference_body_quat_wxyz[:, 0] = sRot.from_euler("z", 30, degrees=True).as_quat()[[3, 0, 1, 2]]
            self.policy.reference_body_quat_wxyz[:, self.policy.anchor_body_index] = sRot.from_euler("z", 50, degrees=True).as_quat()[[3, 0, 1, 2]]
            self.env_data.base_quat = sRot.from_euler("z", 100, degrees=True).as_quat().astype(np.float32)
            self.env_data.torso_quat = sRot.from_euler("z", 125, degrees=True).as_quat().astype(np.float32)

            obs, _ = self.policy.get_observation(self.env_data, Box())
            expected = sRot.from_euler("z", -5, degrees=True).as_matrix()[:, :2].reshape(-1)
            np.testing.assert_allclose(obs[58:64], expected, atol=1e-6)
        finally:
            self.policy.reference_body_quat_wxyz[:] = original_quats

    def test_action_uses_per_joint_scale_and_tracks_raw_action(self):
        obs = np.zeros(154, dtype=np.float32)
        expected_raw = self.policy.session.run(
            ["actions"], {"obs": obs[None, :]}
        )[0].squeeze()

        scaled_action = self.policy.get_action(obs)

        np.testing.assert_allclose(self.policy.last_action, expected_raw)
        np.testing.assert_allclose(
            scaled_action, expected_raw * self.policy.action_scales
        )
        self.assertTrue(np.isfinite(scaled_action).all())

    def test_last_frame_emits_motion_done_once(self):
        self.policy.get_observation(self.env_data, Box())
        self.policy.post_step_callback()

        obs, extras = self.policy.get_observation(self.env_data, Box())
        self.assertEqual(extras["motion_frame"], 1)
        self.assertEqual(extras["CALLBACK"], ["[MOTION_DONE]"])
        np.testing.assert_allclose(obs[:29], np.arange(29) + 30)

        _, extras = self.policy.get_observation(self.env_data, Box())
        self.assertEqual(extras["CALLBACK"], [])

    def test_initial_target_is_first_reference_frame(self):
        np.testing.assert_allclose(
            self.policy.get_init_dof_pos(), np.arange(29, dtype=np.float32)
        )


class TestMjlabLocoActionPipeline(unittest.TestCase):
    def test_indexed_mimic_command_parser(self):
        self.assertEqual(
            MjlabLocoActionPipeline.mimic_index_from_command(
                "[POLICY_MIMIC,2]"
            ),
            2,
        )
        self.assertIsNone(
            MjlabLocoActionPipeline.mimic_index_from_command("[POLICY_MIMIC]")
        )

    def test_full_body_static_override_is_disabled_for_every_state(self):
        state = PolicyInterpManager.InterpState

        self.assertFalse(
            MjlabLocoActionPipeline.transition_override_active(state.IDLE)
        )
        self.assertFalse(
            MjlabLocoActionPipeline.transition_override_active(state.START)
        )
        self.assertFalse(
            MjlabLocoActionPipeline.transition_override_active(state.IN_PROGRESS)
        )
        self.assertFalse(
            MjlabLocoActionPipeline.transition_override_active(state.END)
        )

    def test_locomotion_command_can_be_stopped_before_action(self):
        policy = MjlabVelocityPolicy(
            G1MjlabVelocityPolicyCfg(model_path=LOCO_MODEL_PATH), device="cpu"
        )
        policy.command[:] = [0.5, 0.2, 0.3]

        policy.stop()

        np.testing.assert_allclose(policy.command, 0.0)

    def test_return_blend_moves_from_action_target_to_live_loco_target(self):
        action_target = np.zeros(29, dtype=np.float32)
        loco_target = np.ones(29, dtype=np.float32)

        first = MjlabLocoActionPipeline.blend_return_target(
            action_target, loco_target, step=0, duration=50
        )
        last = MjlabLocoActionPipeline.blend_return_target(
            action_target, loco_target, step=49, duration=50
        )

        np.testing.assert_allclose(first, 0.02)
        np.testing.assert_allclose(last, 1.0)

    def test_indexed_action_switch_selects_requested_policy(self):
        class FakePolicy:
            def __init__(self):
                self.stopped = False
                self.reset_called = False

            def stop(self):
                self.stopped = True

            def reset(self):
                self.reset_called = True

        class FakePolicyManager:
            policy_loco_id = 0
            current_policy_id = 0
            policy_mimic_ids = [1, 2, 3]
            policy_mimic_idx = 0

            def __init__(self):
                self.policies = {index: FakePolicy() for index in range(4)}

            def policy_by_id(self, policy_id):
                return self.policies[policy_id]

            def set_policy(self, policy_id):
                self.current_policy_id = policy_id

        pipeline = object.__new__(MjlabLocoActionPipeline)
        pipeline.control_state = ControlState.LOCO
        pipeline.policy_manager = FakePolicyManager()
        pipeline.return_blend_active = False
        pipeline.policy_locomotion_mimic_flag = 0

        pipeline._switch_to_action(2)

        self.assertTrue(pipeline.policy_manager.policies[0].stopped)
        self.assertTrue(pipeline.policy_manager.policies[3].reset_called)
        self.assertEqual(pipeline.policy_manager.policy_mimic_idx, 2)
        self.assertEqual(pipeline.policy_manager.current_policy_id, 3)
        self.assertEqual(pipeline.policy_locomotion_mimic_flag, 1)


class TestMjlabTrackingConfig(unittest.TestCase):
    def test_combined_config_is_isolated_and_keyboard_only(self):
        from robojudo.config.config_manager import ConfigManager

        cfg = ConfigManager("g1_mjlab_loco_right_overhand").get_cfg()

        self.assertEqual(cfg.pipeline_type, "MjlabLocoActionPipeline")
        self.assertTrue(cfg.do_safety_check)
        self.assertEqual(cfg.loco_policy.policy_type, "MjlabVelocityPolicy")
        self.assertEqual(cfg.loco_policy.policy_name, "locomotion_v3")
        self.assertEqual(len(cfg.mimic_policies), 7)
        self.assertEqual(
            [policy.policy_name for policy in cfg.mimic_policies],
            [
                "right_overhand",
                "back_kick",
                "rear_straight_punch",
                "left_jab",
                "right_cross",
                "left_front_kick",
                "spin_kick",
            ],
        )
        self.assertTrue(
            all(
                policy.policy_type == "MjlabTrackingPolicy"
                for policy in cfg.mimic_policies
            )
        )
        self.assertEqual([ctrl.ctrl_type for ctrl in cfg.ctrl], ["KeyboardCtrl"])
        self.assertEqual(cfg.ctrl[0].triggers["1"], "[POLICY_MIMIC,0]")
        self.assertEqual(cfg.ctrl[0].triggers["2"], "[POLICY_MIMIC,1]")
        self.assertEqual(cfg.ctrl[0].triggers["3"], "[POLICY_MIMIC,2]")
        self.assertEqual(cfg.ctrl[0].triggers["4"], "[POLICY_MIMIC,3]")
        self.assertEqual(cfg.ctrl[0].triggers["5"], "[POLICY_MIMIC,4]")
        self.assertEqual(cfg.ctrl[0].triggers["6"], "[POLICY_MIMIC,5]")
        self.assertEqual(cfg.ctrl[0].triggers["7"], "[POLICY_MIMIC,6]")
        self.assertEqual(cfg.ctrl[0].triggers["9"], "[POLICY_RECOVERY]")
        self.assertEqual(cfg.ctrl[0].triggers["0"], "[POLICY_LOCO]")

    def test_new_leg_action_assets_match_tracking_contract(self):
        from robojudo.config.config_manager import ConfigManager

        cfg = ConfigManager("g1_mjlab_loco_right_overhand").get_cfg()
        policies = {policy.policy_name: policy for policy in cfg.mimic_policies}

        for name in ("left_front_kick", "spin_kick"):
            policy = MjlabTrackingPolicy(policies[name], device="cpu")
            self.assertEqual(policy.observation_size, 154)
            self.assertEqual(policy.num_actions, 29)
            self.assertEqual(policy.reference_joint_pos.shape[1], 29)
            self.assertGreater(policy.motion_frame_count, 1)


if __name__ == "__main__":
    unittest.main()
