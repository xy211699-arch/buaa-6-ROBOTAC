import unittest

import numpy as np
from box import Box

from robojudo.config.g1.policy.g1_amp_recovery_policy_cfg import (
    G1AmpRecoveryPolicyCfg,
)
from robojudo.policy.amp_recovery_policy import AmpRecoveryPolicy


MODEL_PATH = (
    "/root/gpufree-data/wbc_fsm/model/loco/"
    "Unitree-G1-AMP-Flat_model_30000.onnx"
)


class TestAmpRecoveryPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = AmpRecoveryPolicy(
            G1AmpRecoveryPolicyCfg(model_path=MODEL_PATH), device="cpu"
        )

    def setUp(self):
        self.policy.reset()
        self.dof_delta = np.arange(29, dtype=np.float32) * 0.01
        self.dof_vel = np.arange(29, dtype=np.float32) * 0.02
        self.env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "base_ang_vel": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "dof_pos": self.policy.default_dof_pos + self.dof_delta,
                "dof_vel": self.dof_vel,
            }
        )

    def test_model_and_control_contract(self):
        self.assertEqual(self.policy.frame_observation_size, 96)
        self.assertEqual(self.policy.observation_size, 384)
        self.assertEqual(self.policy.history_length, 4)
        self.assertEqual(self.policy.num_actions, 29)
        self.assertEqual(self.policy.action_scales.shape, (29,))
        self.assertEqual(self.policy.cfg_action_dof.torque_limits[0], 139.0)
        self.assertEqual(self.policy.cfg_action_dof.torque_limits[20], 10.0)

    def test_first_observation_repeats_current_state_four_times(self):
        obs, extras = self.policy.get_observation(self.env_data, Box())

        expected_frame = np.concatenate(
            [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                self.dof_delta,
                self.dof_vel,
                np.zeros(29, dtype=np.float32),
            ]
        )

        self.assertEqual(obs.shape, (384,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(
            obs.reshape(4, 96), np.tile(expected_frame, (4, 1)), atol=1e-6
        )
        np.testing.assert_allclose(extras["commands"], 0.0)

    def test_history_appends_previous_raw_action(self):
        self.policy.get_observation(self.env_data, Box())
        previous_action = np.arange(29, dtype=np.float32) * 0.1
        self.policy.last_action = previous_action.copy()

        obs, _ = self.policy.get_observation(self.env_data, Box())
        frames = obs.reshape(4, 96)

        np.testing.assert_allclose(frames[:3, -29:], 0.0)
        np.testing.assert_allclose(frames[3, -29:], previous_action)

    def test_action_uses_per_joint_scale_and_tracks_raw_action(self):
        obs = np.zeros(384, dtype=np.float32)
        expected_raw = self.policy.session.run(
            ["actions"], {"obs": obs[None, :]}
        )[0].squeeze()

        scaled_action = self.policy.get_action(obs)

        np.testing.assert_allclose(self.policy.last_action, expected_raw)
        np.testing.assert_allclose(
            scaled_action, expected_raw * self.policy.action_scales
        )
        self.assertEqual(scaled_action.shape, (29,))
        self.assertTrue(np.isfinite(scaled_action).all())


if __name__ == "__main__":
    unittest.main()