import unittest

import numpy as np
from box import Box

from robojudo.config.g1.policy.g1_mjlab_velocity_policy_cfg import (
    G1MjlabVelocityPolicyCfg,
)
from robojudo.policy.mjlab_velocity_policy import MjlabVelocityPolicy


MODEL_PATH = (
    "/root/gpufree-data/RoboJuDo/assets/models/g1/mjlab/"
    "locomotion_v3/policy.onnx"
)


class TestMjlabVelocityPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = MjlabVelocityPolicy(
            cfg_policy=G1MjlabVelocityPolicyCfg(model_path=MODEL_PATH),
            device="cpu",
        )

    def setUp(self):
        self.policy.reset()
        self.env_data = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "base_ang_vel": np.zeros(3),
                "dof_pos": self.policy.default_dof_pos.copy(),
                "dof_vel": np.zeros(29),
            }
        )

    @staticmethod
    def keyboard_event(name, pressed=True):
        return Box(
            {
                "KeyboardCtrl": {
                    "keyboard_event": [
                        {"type": "keyboard", "name": name, "pressed": pressed}
                    ]
                }
            }
        )

    def test_model_contract_and_metadata(self):
        self.assertEqual(self.policy.num_dofs, 29)
        self.assertEqual(self.policy.num_actions, 29)
        self.assertEqual(self.policy.observation_size, 98)
        self.assertEqual(self.policy.action_scales.shape, (29,))
        self.assertEqual(
            self.policy.cfg_action_dof.joint_names[0], "left_hip_pitch_joint"
        )
        self.assertEqual(
            self.policy.cfg_action_dof.joint_names[-1], "right_wrist_yaw_joint"
        )

    def test_zero_command_observation_order(self):
        obs, extras = self.policy.get_observation(self.env_data, Box())

        self.assertEqual(obs.shape, (98,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[0:3], 0.0)
        np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(obs[6:9], 0.0)
        np.testing.assert_allclose(obs[9:11], 0.0)
        np.testing.assert_allclose(obs[11:40], 0.0, atol=1e-6)
        np.testing.assert_allclose(obs[40:69], 0.0)
        np.testing.assert_allclose(obs[69:98], 0.0)
        np.testing.assert_allclose(extras["commands"], 0.0)

    def test_keyboard_command_latches_until_space(self):
        obs, _ = self.policy.get_observation(
            self.env_data, self.keyboard_event("w")
        )
        np.testing.assert_allclose(obs[6:9], [0.5, 0.0, 0.0])

        obs, _ = self.policy.get_observation(self.env_data, Box())
        np.testing.assert_allclose(obs[6:9], [0.5, 0.0, 0.0])

        obs, _ = self.policy.get_observation(
            self.env_data, self.keyboard_event("Key.space")
        )
        np.testing.assert_allclose(obs[6:9], 0.0)

    def test_phase_advances_but_is_hidden_while_standing(self):
        self.policy.post_step_callback()
        obs, _ = self.policy.get_observation(self.env_data, Box())
        np.testing.assert_allclose(obs[9:11], 0.0)

        obs, _ = self.policy.get_observation(
            self.env_data, self.keyboard_event("w")
        )
        phase = self.policy.dt / self.policy.gait_period
        np.testing.assert_allclose(
            obs[9:11],
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            atol=1e-6,
        )

    def test_action_uses_per_joint_scale_and_tracks_raw_action(self):
        obs = np.zeros(98, dtype=np.float32)
        expected_raw = self.policy.session.run(
            ["actions"], {"obs": obs[None, :]}
        )[0].squeeze()

        scaled_action = self.policy.get_action(obs)

        np.testing.assert_allclose(self.policy.last_action, expected_raw)
        np.testing.assert_allclose(
            scaled_action, expected_raw * self.policy.action_scales
        )
        self.assertTrue(np.isfinite(scaled_action).all())


class TestMjlabVelocityConfig(unittest.TestCase):
    def test_dedicated_g1_config_uses_keyboard_and_mjlab_policy(self):
        from robojudo.config.config_manager import ConfigManager

        cfg = ConfigManager("g1_mjlab_loco").get_cfg()

        self.assertEqual(cfg.pipeline_type, "RlPipeline")
        self.assertEqual(cfg.policy.policy_type, "MjlabVelocityPolicy")
        self.assertEqual(cfg.policy.policy_name, "locomotion_v3")
        self.assertTrue(
            cfg.policy.policy_file.endswith("/locomotion_v3/policy.onnx")
        )
        self.assertEqual([ctrl.ctrl_type for ctrl in cfg.ctrl], ["KeyboardCtrl"])


if __name__ == "__main__":
    unittest.main()
