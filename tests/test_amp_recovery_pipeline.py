import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from box import Box

from robojudo.pipeline.mjlab_loco_action_pipeline import (
    ControlState,
    MjlabLocoActionPipeline,
)
from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline


class FakePolicy:
    def __init__(self):
        self.reset_called = False
        self.stopped = False
        self.cfg_action_dof = object()

    def reset(self):
        self.reset_called = True

    def stop(self):
        self.stopped = True

    def post_step_callback(self, commands=None):
        return

    def debug_viz(self, visualizer, env_data, ctrl_data, extras):
        return


class FakePolicyManager:
    policy_loco_id = 0

    def __init__(self, current_policy_id=0):
        self.current_policy_id = current_policy_id
        self.policy_mimic_idx = 2
        self.policy_mimic_ids = [1, 2, 3, 4]
        self.policies = {idx: FakePolicy() for idx in range(5)}

    @property
    def policy(self):
        return self.policies[self.current_policy_id]

    def policy_by_id(self, policy_id):
        return self.policies[policy_id]

    def set_policy(self, policy_id):
        self.current_policy_id = policy_id

    def step(self, env_data, ctrl_data):
        return


class FakeSwitchEnv:
    def __init__(self):
        self.override_cfg = None

    def update_dof_cfg(self, override_cfg=None):
        self.override_cfg = override_cfg


class FakeCtrlManager:
    def post_step_callback(self, ctrl_data):
        return


class TestAmpRecoveryPipeline(unittest.TestCase):
    def make_pipeline(self, state=ControlState.LOCO, current_policy_id=0):
        pipeline = object.__new__(MjlabLocoActionPipeline)
        pipeline.control_state = state
        pipeline.policy_manager = FakePolicyManager(current_policy_id)
        pipeline.recovery_policy = FakePolicy()
        pipeline.env = FakeSwitchEnv()
        pipeline.return_blend_active = False
        pipeline.return_blend_step = 0
        pipeline.return_blend_start = None
        pipeline.policy_locomotion_mimic_flag = 0
        pipeline.recovery_upright_count = 0
        pipeline.recovery_stable_steps = 25
        pipeline.recovery_upright_angle = 0.35
        pipeline.recovery_upright_height = 0.65
        pipeline.stabilize_max_base_lin_vel = 0.3
        pipeline.stabilize_max_base_ang_vel = 0.5
        pipeline.action3_disturbance_guard_enabled = True
        pipeline.action3_disturbance_guard_grace_steps = 10
        pipeline.action3_disturbance_guard_tilt_angle = 0.35
        pipeline.action3_disturbance_guard_ang_vel = 3.5
        pipeline.action3_disturbance_guard_trigger_steps = 3
        pipeline.action3_disturbance_guard_elapsed_steps = 0
        pipeline.action3_disturbance_guard_unstable_steps = 0
        pipeline.timestep = 0
        pipeline.ctrl_manager = FakeCtrlManager()
        pipeline.visualizer = None
        pipeline.do_safety_check = False
        pipeline.cfg = Box({"debug": {"log_obs": False}})
        return pipeline

    def test_guard_config_is_ready_before_base_pipeline_self_check(self):
        cfg = Box(
            {
                "action3_disturbance_guard_enabled": True,
                "action3_disturbance_guard_grace_steps": 10,
                "action3_disturbance_guard_tilt_angle": 0.35,
                "action3_disturbance_guard_ang_vel": 3.5,
                "action3_disturbance_guard_trigger_steps": 3,
            }
        )
        test_case = self

        def base_init(pipeline, _cfg):
            test_case.assertTrue(pipeline.action3_disturbance_guard_enabled)
            test_case.assertEqual(
                pipeline.action3_disturbance_guard_elapsed_steps, 0
            )
            raise RuntimeError("base initialization reached")

        with patch.object(RlLocoMimicPipeline, "__init__", new=base_init):
            with self.assertRaisesRegex(RuntimeError, "base initialization reached"):
                MjlabLocoActionPipeline(cfg)

    @staticmethod
    def action_env_data(tilt=0.0, roll_rate=0.0):
        return Box(
            {
                "base_quat": np.array(
                    [np.sin(tilt / 2.0), 0.0, 0.0, np.cos(tilt / 2.0)]
                ),
                "base_ang_vel": np.array([roll_rate, 0.0, 0.0]),
            }
        )

    def test_action_three_guard_can_be_disabled_to_preserve_old_behavior(self):
        pipeline = self.make_pipeline(ControlState.ACTION, current_policy_id=3)
        pipeline.action3_disturbance_guard_enabled = False
        disturbed = self.action_env_data(tilt=0.5, roll_rate=2.0)

        for _ in range(20):
            self.assertFalse(pipeline._update_action3_disturbance_guard(disturbed))

        self.assertEqual(pipeline.action3_disturbance_guard_unstable_steps, 0)

    def test_action_three_guard_uses_grace_and_consecutive_unstable_steps(self):
        pipeline = self.make_pipeline(ControlState.ACTION, current_policy_id=3)
        pipeline.action3_disturbance_guard_grace_steps = 2
        disturbed = self.action_env_data(tilt=0.4)

        triggered = [
            pipeline._update_action3_disturbance_guard(disturbed)
            for _ in range(5)
        ]

        self.assertEqual(triggered, [False, False, False, False, True])

    def test_action_three_guard_ignores_normal_action_dynamic_envelope(self):
        pipeline = self.make_pipeline(ControlState.ACTION, current_policy_id=3)
        pipeline.action3_disturbance_guard_grace_steps = 0
        normal_action = self.action_env_data(tilt=0.26, roll_rate=2.8)

        for _ in range(3):
            self.assertFalse(
                pipeline._update_action3_disturbance_guard(normal_action)
            )

        self.assertEqual(pipeline.action3_disturbance_guard_unstable_steps, 0)

    def test_action_three_guard_resets_after_a_stable_sample(self):
        pipeline = self.make_pipeline(ControlState.ACTION, current_policy_id=3)
        pipeline.action3_disturbance_guard_grace_steps = 0
        disturbed = self.action_env_data(roll_rate=3.6)
        stable = self.action_env_data(tilt=0.2, roll_rate=0.5)

        self.assertFalse(pipeline._update_action3_disturbance_guard(disturbed))
        self.assertFalse(pipeline._update_action3_disturbance_guard(disturbed))
        self.assertFalse(pipeline._update_action3_disturbance_guard(stable))
        self.assertFalse(pipeline._update_action3_disturbance_guard(disturbed))
        self.assertFalse(pipeline._update_action3_disturbance_guard(disturbed))
        self.assertTrue(pipeline._update_action3_disturbance_guard(disturbed))

    def test_action_three_guard_switches_to_stabilize_but_ignores_other_actions(self):
        disturbed = self.action_env_data(tilt=0.4)
        pd_target = np.zeros(29, dtype=np.float32)

        action_three = self.make_pipeline(ControlState.ACTION, current_policy_id=3)
        action_three.action3_disturbance_guard_grace_steps = 0
        action_three.action3_disturbance_guard_trigger_steps = 1
        action_three.post_step_callback(
            disturbed, {"COMMANDS": []}, {"CALLBACK": []}, pd_target
        )
        self.assertEqual(action_three.control_state, ControlState.STABILIZE)

        action_one = self.make_pipeline(ControlState.ACTION, current_policy_id=1)
        action_one.policy_manager.policy_mimic_idx = 0
        action_one.action3_disturbance_guard_grace_steps = 0
        action_one.action3_disturbance_guard_trigger_steps = 1
        action_one.post_step_callback(
            disturbed, {"COMMANDS": []}, {"CALLBACK": []}, pd_target
        )
        self.assertEqual(action_one.control_state, ControlState.ACTION)

    def test_only_action_three_routes_to_stabilize_after_completion(self):
        route = getattr(MjlabLocoActionPipeline, "_motion_done_command", None)
        self.assertIsNotNone(route)
        if route is None:
            return
        self.assertEqual(route(2), "[POLICY_STABILIZE]")
        self.assertEqual(route(4), "[POLICY_LOCO]")
        self.assertEqual(route(5), "[POLICY_LOCO]")
        self.assertEqual(route(6), "[POLICY_LOCO]")
        self.assertEqual(route(0), "[POLICY_LOCO]")
        self.assertEqual(route(1), "[POLICY_LOCO]")
        self.assertEqual(route(3), "[POLICY_LOCO]")

    def test_switch_to_stabilize_uses_amp_policy(self):
        stabilize_state = ControlState.__members__.get("STABILIZE")
        self.assertIsNotNone(stabilize_state)
        if stabilize_state is None:
            return
        pipeline = self.make_pipeline(ControlState.ACTION, current_policy_id=3)

        pipeline._switch_to_stabilize()

        self.assertEqual(pipeline.control_state, stabilize_state)
        self.assertEqual(pipeline.policy_manager.current_policy_id, 0)
        self.assertTrue(pipeline.recovery_policy.reset_called)
        self.assertIs(
            pipeline.env.override_cfg, pipeline.recovery_policy.cfg_action_dof
        )

    def test_stabilize_requires_low_linear_and_angular_velocity(self):
        stabilize_state = ControlState.__members__.get("STABILIZE")
        self.assertIsNotNone(stabilize_state)
        if stabilize_state is None:
            return
        pipeline = self.make_pipeline(stabilize_state)
        fast = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "base_pos": np.array([0.0, 0.0, 0.75]),
                "base_lin_vel": np.array([0.31, 0.0, 0.0]),
                "base_ang_vel": np.array([0.0, 0.0, 0.51]),
            }
        )
        for _ in range(25):
            pipeline._update_recovery_stability(fast)
        self.assertFalse(pipeline.recovery_ready)

        still = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "base_pos": np.array([0.0, 0.0, 0.75]),
                "base_lin_vel": np.zeros(3),
                "base_ang_vel": np.zeros(3),
            }
        )
        for _ in range(25):
            pipeline._update_recovery_stability(still)
        self.assertTrue(pipeline.recovery_ready)

    def test_stabilize_returns_to_loco_automatically_when_ready(self):
        stabilize_state = ControlState.__members__.get("STABILIZE")
        self.assertIsNotNone(stabilize_state)
        if stabilize_state is None:
            return
        pipeline = self.make_pipeline(stabilize_state)
        pipeline.recovery_upright_count = 24
        stable = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "base_pos": np.array([0.0, 0.0, 0.75]),
                "base_lin_vel": np.zeros(3),
                "base_ang_vel": np.zeros(3),
            }
        )

        pipeline.post_step_callback(
            stable,
            {"COMMANDS": []},
            {"CALLBACK": []},
            np.zeros(29, dtype=np.float32),
        )

        self.assertEqual(pipeline.control_state, ControlState.RETURN)
        self.assertTrue(pipeline.return_blend_active)

    def test_switch_to_recovery_cancels_action_and_uses_recovery_gains(self):
        pipeline = self.make_pipeline(ControlState.ACTION, current_policy_id=2)
        pipeline.return_blend_active = True

        pipeline._switch_to_recovery()

        self.assertEqual(pipeline.control_state, ControlState.RECOVERY)
        self.assertEqual(pipeline.policy_manager.current_policy_id, 0)
        self.assertTrue(pipeline.policy_manager.policies[0].stopped)
        self.assertTrue(pipeline.recovery_policy.reset_called)
        self.assertIs(
            pipeline.env.override_cfg, pipeline.recovery_policy.cfg_action_dof
        )
        self.assertFalse(pipeline.return_blend_active)

    def test_recovery_exit_is_blocked_until_upright_is_stable(self):
        pipeline = self.make_pipeline(ControlState.RECOVERY)
        recovery_target = np.arange(29, dtype=np.float32)

        pipeline.recovery_upright_count = 24
        pipeline._switch_to_loco(recovery_target)
        self.assertEqual(pipeline.control_state, ControlState.RECOVERY)
        self.assertFalse(pipeline.return_blend_active)

        pipeline.recovery_upright_count = 25
        pipeline._switch_to_loco(recovery_target)
        self.assertEqual(pipeline.control_state, ControlState.RETURN)
        self.assertTrue(pipeline.return_blend_active)
        np.testing.assert_allclose(pipeline.return_blend_start, recovery_target)

    def test_upright_counter_requires_height_and_tilt_for_25_steps(self):
        pipeline = self.make_pipeline(ControlState.RECOVERY)
        upright = Box(
            {
                "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "base_pos": np.array([0.0, 0.0, 0.75]),
            }
        )

        for _ in range(25):
            pipeline._update_recovery_stability(upright)
        self.assertTrue(pipeline.recovery_ready)

        tilted = Box(
            {
                "base_quat": np.array([0.70710678, 0.0, 0.0, 0.70710678]),
                "base_pos": np.array([0.0, 0.0, 0.75]),
            }
        )
        pipeline._update_recovery_stability(tilted)
        self.assertFalse(pipeline.recovery_ready)
        self.assertEqual(pipeline.recovery_upright_count, 0)

    def test_sim_fall_is_latched_without_reborn(self):
        class FakeSimEnv:
            cfg_env = Box({"is_sim": True})
            base_quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])

            def __init__(self):
                self.reborn_count = 0

            def reborn(self):
                self.reborn_count += 1

        pipeline = object.__new__(MjlabLocoActionPipeline)
        pipeline.do_safety_check = True
        pipeline.env = FakeSimEnv()
        pipeline.fallen = False
        pipeline._fall_warning_emitted = False

        pipeline.safety_check()

        self.assertTrue(pipeline.fallen)
        self.assertEqual(pipeline.env.reborn_count, 0)

    def test_real_fall_keeps_shutdown_behavior(self):
        class FakeRealEnv:
            cfg_env = Box({"is_sim": False})
            base_quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])

            def __init__(self):
                self.shutdown_count = 0

            def shutdown(self):
                self.shutdown_count += 1

        pipeline = object.__new__(MjlabLocoActionPipeline)
        pipeline.do_safety_check = True
        pipeline.env = FakeRealEnv()
        pipeline.fallen = False
        pipeline._fall_warning_emitted = False

        pipeline.safety_check()

        self.assertEqual(pipeline.env.shutdown_count, 1)


class TestAmpRecoveryConfig(unittest.TestCase):
    def test_combined_config_registers_recovery_without_changing_actions(self):
        from robojudo.config.config_manager import ConfigManager

        cfg = ConfigManager("g1_mjlab_loco_right_overhand").get_cfg()

        self.assertEqual(
            cfg.ctrl[0].triggers,
            {
                "Key.esc": "[SHUTDOWN]",
                "`": "[SIM_REBORN]",
                "1": "[POLICY_MIMIC,0]",
                "2": "[POLICY_MIMIC,1]",
                "3": "[POLICY_MIMIC,2]",
                "4": "[POLICY_MIMIC,3]",
                "5": "[POLICY_MIMIC,4]",
                "6": "[POLICY_MIMIC,5]",
                "7": "[POLICY_MIMIC,6]",
                "9": "[POLICY_RECOVERY]",
                "0": "[POLICY_LOCO]",
            },
        )
        self.assertEqual(cfg.recovery_policy.policy_type, "AmpRecoveryPolicy")
        self.assertEqual(cfg.recovery_policy.policy_name, "amp_recovery")
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
        right_cross = cfg.mimic_policies[4]
        self.assertTrue(Path(right_cross.policy_file).is_file())
        self.assertTrue(Path(right_cross.motion_file).is_file())
        self.assertEqual(cfg.recovery_upright_angle, 0.35)
        self.assertEqual(cfg.recovery_upright_height, 0.65)
        self.assertEqual(cfg.recovery_stable_steps, 25)
        self.assertEqual(cfg.stabilize_max_base_lin_vel, 0.3)
        self.assertEqual(cfg.stabilize_max_base_ang_vel, 0.5)
        self.assertTrue(cfg.action3_disturbance_guard_enabled)
        self.assertEqual(cfg.action3_disturbance_guard_grace_steps, 10)
        self.assertEqual(cfg.action3_disturbance_guard_tilt_angle, 0.35)
        self.assertEqual(cfg.action3_disturbance_guard_ang_vel, 3.5)
        self.assertEqual(cfg.action3_disturbance_guard_trigger_steps, 3)

        post_action_only = ConfigManager(
            "g1_mjlab_loco_right_overhand_post_action_only"
        ).get_cfg()
        self.assertFalse(post_action_only.action3_disturbance_guard_enabled)


if __name__ == "__main__":
    unittest.main()