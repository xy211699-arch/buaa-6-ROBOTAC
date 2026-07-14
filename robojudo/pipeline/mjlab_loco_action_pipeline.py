import logging
from enum import Enum

import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_loco_mimic_pipeline import RlLocoMimicPipeline
from robojudo.pipeline.rl_pipeline import PolicyWrapper, RlPipeline
from robojudo.utils.util_func import get_gravity_orientation


logger = logging.getLogger(__name__)


class ControlState(Enum):
    LOCO = "loco"
    ACTION = "action"
    RETURN = "return"
    STABILIZE = "stabilize"
    RECOVERY = "recovery"


@pipeline_registry.register
class MjlabLocoActionPipeline(RlLocoMimicPipeline):
    """Direct action entry, AMP recovery, and a dynamic return to locomotion."""

    def __init__(self, cfg):
        self.control_state = ControlState.LOCO
        self.recovery_policy = None
        self.return_blend_duration = 50
        self.return_blend_active = False
        self.return_blend_step = 0
        self.return_blend_start = None
        self.fallen = False
        self._fall_warning_emitted = False
        self.action3_disturbance_guard_enabled = (
            cfg.action3_disturbance_guard_enabled
        )
        self.action3_disturbance_guard_grace_steps = (
            cfg.action3_disturbance_guard_grace_steps
        )
        self.action3_disturbance_guard_tilt_angle = (
            cfg.action3_disturbance_guard_tilt_angle
        )
        self.action3_disturbance_guard_ang_vel = (
            cfg.action3_disturbance_guard_ang_vel
        )
        self.action3_disturbance_guard_trigger_steps = (
            cfg.action3_disturbance_guard_trigger_steps
        )
        self.action3_disturbance_guard_elapsed_steps = 0
        self.action3_disturbance_guard_unstable_steps = 0

        super().__init__(cfg)

        self.recovery_policy = PolicyWrapper(
            cfg_policy=self.cfg.recovery_policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )
        if self.recovery_policy.freq != self.freq:
            raise ValueError(
                "Recovery and locomotion policies must use the same frequency"
            )

        self.recovery_upright_angle = self.cfg.recovery_upright_angle
        self.recovery_upright_height = self.cfg.recovery_upright_height
        self.recovery_stable_steps = self.cfg.recovery_stable_steps
        self.stabilize_max_base_lin_vel = self.cfg.stabilize_max_base_lin_vel
        self.stabilize_max_base_ang_vel = self.cfg.stabilize_max_base_ang_vel
        self.recovery_upright_count = 0

        loco_policy = self.policy_manager.policy_by_id(
            self.policy_manager.policy_loco_id
        )
        self.loco_dof_pos = loco_policy.get_init_dof_pos()
        self.policy_manager.loco_dof_pos = self.loco_dof_pos.copy()
        self.policy_manager.override_dof_pos = self.loco_dof_pos.copy()

    @property
    def policy(self):
        if (
            self.control_state in (ControlState.RECOVERY, ControlState.STABILIZE)
            and self.recovery_policy is not None
        ):
            return self.recovery_policy
        return self.policy_manager.policy

    @property
    def recovery_ready(self):
        return self.recovery_upright_count >= self.recovery_stable_steps

    @staticmethod
    def transition_override_active(interp_state):
        return False

    @staticmethod
    def blend_return_target(action_target, loco_target, step, duration):
        alpha = min((step + 1) / duration, 1.0)
        return (1.0 - alpha) * action_target + alpha * loco_target

    @staticmethod
    def mimic_index_from_command(command):
        prefix = "[POLICY_MIMIC,"
        if not command.startswith(prefix) or not command.endswith("]"):
            return None
        return int(command[len(prefix) : -1])

    @staticmethod
    def _motion_done_command(mimic_idx):
        if mimic_idx == 2:
            return "[POLICY_STABILIZE]"
        return "[POLICY_LOCO]"

    @staticmethod
    def _tilt_angle(base_quat):
        gravity_ori = get_gravity_orientation(base_quat)
        return float(np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0)))

    def _reset_action3_disturbance_guard(self):
        self.action3_disturbance_guard_elapsed_steps = 0
        self.action3_disturbance_guard_unstable_steps = 0

    def _update_action3_disturbance_guard(self, env_data):
        is_guarded_action = (
            self.action3_disturbance_guard_enabled
            and self.control_state == ControlState.ACTION
            and self.policy_manager.policy_mimic_idx == 2
        )
        if not is_guarded_action:
            self._reset_action3_disturbance_guard()
            return False

        self.action3_disturbance_guard_elapsed_steps += 1
        if (
            self.action3_disturbance_guard_elapsed_steps
            <= self.action3_disturbance_guard_grace_steps
        ):
            self.action3_disturbance_guard_unstable_steps = 0
            return False

        base_quat = np.asarray(env_data.base_quat, dtype=np.float32)
        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
        if (
            base_quat.shape != (4,)
            or base_ang_vel.shape != (3,)
            or not np.isfinite(base_quat).all()
            or not np.isfinite(base_ang_vel).all()
        ):
            self.action3_disturbance_guard_unstable_steps = 0
            return False

        tilt = self._tilt_angle(base_quat)
        horizontal_ang_vel = float(np.linalg.norm(base_ang_vel[:2]))
        unstable = (
            tilt > self.action3_disturbance_guard_tilt_angle
            or horizontal_ang_vel > self.action3_disturbance_guard_ang_vel
        )
        if unstable:
            self.action3_disturbance_guard_unstable_steps = min(
                self.action3_disturbance_guard_unstable_steps + 1,
                self.action3_disturbance_guard_trigger_steps,
            )
        else:
            self.action3_disturbance_guard_unstable_steps = 0

        triggered = (
            self.action3_disturbance_guard_unstable_steps
            >= self.action3_disturbance_guard_trigger_steps
        )
        if triggered:
            logger.warning(
                "Action 3 disturbance guard triggered: tilt=%.3f rad, "
                "horizontal angular speed=%.3f rad/s",
                tilt,
                horizontal_ang_vel,
            )
        return triggered

    def _update_recovery_stability(self, env_data):
        recovery_states = (ControlState.RECOVERY, ControlState.STABILIZE)
        if self.control_state not in recovery_states:
            self.recovery_upright_count = 0
            return

        base_pos = env_data.base_pos
        if base_pos is None:
            self.recovery_upright_count = 0
            return

        base_pos = np.asarray(base_pos, dtype=np.float32)
        base_quat = np.asarray(env_data.base_quat, dtype=np.float32)
        stable = (
            base_pos.shape == (3,)
            and base_quat.shape == (4,)
            and np.isfinite(base_pos).all()
            and np.isfinite(base_quat).all()
            and base_pos[2] >= self.recovery_upright_height
            and self._tilt_angle(base_quat) <= self.recovery_upright_angle
        )
        if stable and self.control_state == ControlState.STABILIZE:
            base_lin_vel = env_data.base_lin_vel
            base_ang_vel = env_data.base_ang_vel
            if base_lin_vel is None or base_ang_vel is None:
                stable = False
            else:
                base_lin_vel = np.asarray(base_lin_vel, dtype=np.float32)
                base_ang_vel = np.asarray(base_ang_vel, dtype=np.float32)
                stable = (
                    base_lin_vel.shape == (3,)
                    and base_ang_vel.shape == (3,)
                    and np.isfinite(base_lin_vel).all()
                    and np.isfinite(base_ang_vel).all()
                    and np.linalg.norm(base_lin_vel)
                    <= self.stabilize_max_base_lin_vel
                    and np.linalg.norm(base_ang_vel)
                    <= self.stabilize_max_base_ang_vel
                )

        if stable:
            self.recovery_upright_count = min(
                self.recovery_upright_count + 1, self.recovery_stable_steps
            )
        else:
            self.recovery_upright_count = 0

    def _switch_to_action(self, mimic_idx=None):
        if (
            self.control_state != ControlState.LOCO
            or self.policy_manager.current_policy_id
            != self.policy_manager.policy_loco_id
            or self.return_blend_active
        ):
            return

        if mimic_idx is not None:
            if not 0 <= mimic_idx < len(self.policy_manager.policy_mimic_ids):
                raise ValueError(f"Invalid mimic policy index: {mimic_idx}")
            self.policy_manager.policy_mimic_idx = mimic_idx

        self._reset_action3_disturbance_guard()
        loco_policy = self.policy_manager.policy_by_id(
            self.policy_manager.policy_loco_id
        )
        loco_policy.stop()

        action_policy_id = self.policy_manager.policy_mimic_ids[
            self.policy_manager.policy_mimic_idx
        ]
        self.policy_manager.policy_by_id(action_policy_id).reset()
        self.policy_manager.set_policy(action_policy_id)
        self.policy_locomotion_mimic_flag = 1
        self.control_state = ControlState.ACTION

    def _enter_amp_state(self, target_state):
        if self.control_state == target_state:
            return

        self._reset_action3_disturbance_guard()
        self.return_blend_active = False
        self.return_blend_step = 0
        self.return_blend_start = None

        loco_policy_id = self.policy_manager.policy_loco_id
        if self.policy_manager.current_policy_id != loco_policy_id:
            self.policy_manager.set_policy(loco_policy_id)
        loco_policy = self.policy_manager.policy_by_id(loco_policy_id)
        loco_policy.stop()

        self.recovery_policy.reset()
        self.env.update_dof_cfg(
            override_cfg=self.recovery_policy.cfg_action_dof
        )
        self.policy_locomotion_mimic_flag = 0
        self.recovery_upright_count = 0
        self.control_state = target_state
        logger.warning(
            "Entered AMP %s state with zero velocity command",
            target_state.value,
        )

    def _switch_to_recovery(self):
        self._enter_amp_state(ControlState.RECOVERY)

    def _switch_to_stabilize(self):
        self._enter_amp_state(ControlState.STABILIZE)

    def _switch_to_loco(self, action_pd_target):
        if self.control_state in (ControlState.RECOVERY, ControlState.STABILIZE):
            if not self.recovery_ready:
                logger.warning("Recovery exit ignored: robot is not stably upright")
                return
        elif (
            self.policy_manager.current_policy_id
            == self.policy_manager.policy_loco_id
        ):
            return

        loco_policy = self.policy_manager.policy_by_id(
            self.policy_manager.policy_loco_id
        )
        loco_policy.reset()
        self.return_blend_start = np.asarray(
            action_pd_target, dtype=np.float32
        ).copy()
        self.return_blend_step = 0
        self.return_blend_active = True
        self.policy_manager.set_policy(self.policy_manager.policy_loco_id)
        self.policy_locomotion_mimic_flag = 0
        self.recovery_upright_count = 0
        self.control_state = ControlState.RETURN

    def _reset_to_loco(self):
        self.return_blend_active = False
        self.return_blend_step = 0
        self.return_blend_start = None
        self.recovery_upright_count = 0
        self.fallen = False
        self._fall_warning_emitted = False

        loco_policy_id = self.policy_manager.policy_loco_id
        loco_policy = self.policy_manager.policy_by_id(loco_policy_id)
        loco_policy.reset()
        if self.policy_manager.current_policy_id != loco_policy_id:
            self.policy_manager.set_policy(loco_policy_id)
        else:
            self.env.update_dof_cfg(override_cfg=loco_policy.cfg_action_dof)
        self.recovery_policy.reset()
        self.policy_locomotion_mimic_flag = 0
        self.control_state = ControlState.LOCO

    def safety_check(self):
        if not self.do_safety_check:
            return

        angle = self._tilt_angle(self.env.base_quat)
        is_fallen = abs(angle) > 1.0
        if self.env.cfg_env.is_sim and is_fallen:
            self.fallen = True
            if not self._fall_warning_emitted:
                logger.warning(
                    "Robot fallen in simulation; automatic reborn is suppressed"
                )
                self._fall_warning_emitted = True
            return

        self.fallen = is_fallen
        if not is_fallen:
            self._fall_warning_emitted = False
        RlPipeline.safety_check(self)

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = list(ctrl_data.get("COMMANDS", []))
        active_policy = self.policy
        self._update_recovery_stability(env_data)
        disturbance_guard_triggered = (
            self._update_action3_disturbance_guard(env_data)
        )
        if disturbance_guard_triggered:
            commands.append("[POLICY_STABILIZE]")

        for callback in extras.get("CALLBACK", []):
            if (
                callback == "[MOTION_DONE]"
                and self.control_state == ControlState.ACTION
                and not disturbance_guard_triggered
            ):
                commands.append(
                    self._motion_done_command(self.policy_manager.policy_mimic_idx)
                )

        if self.control_state == ControlState.STABILIZE and self.recovery_ready:
            commands.append("[POLICY_LOCO]")

        for command in commands:
            if command == "[SHUTDOWN]":
                self.env.shutdown()
            elif command == "[SIM_REBORN]":
                if hasattr(self.env, "reborn"):
                    self._reset_to_loco()
                    self.env.reborn()
                    active_policy.reset_alignment()
            elif command.startswith("[POLICY_SWITCH]"):
                if self.control_state in (ControlState.RECOVERY, ControlState.STABILIZE):
                    continue
                switch_target = command.split(",")[1]
                if switch_target == "NEXT":
                    self.policy_manager.toggle_mimic_policy(1)
                elif switch_target == "LAST":
                    self.policy_manager.toggle_mimic_policy(-1)
            elif command == "[POLICY_LOCO]":
                self._switch_to_loco(pd_target)
            elif command == "[POLICY_RECOVERY]":
                self._switch_to_recovery()
            elif command == "[POLICY_STABILIZE]":
                self._switch_to_stabilize()
            elif command.startswith("[POLICY_MIMIC,"):
                self._switch_to_action(self.mimic_index_from_command(command))
            elif command == "[POLICY_MIMIC]":
                self._switch_to_action()

        self.ctrl_manager.post_step_callback(ctrl_data)
        active_policy.post_step_callback(commands)
        if self.visualizer is not None:
            active_policy.debug_viz(
                self.visualizer, env_data, ctrl_data, extras
            )

        self.policy_manager.step(env_data, ctrl_data)
        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        current_is_loco = (
            self.control_state in (ControlState.LOCO, ControlState.RETURN)
            and self.policy_manager.current_policy_id
            == self.policy_manager.policy_loco_id
        )
        if current_is_loco:
            ctrl_data["ref_dof_pos"] = self.policy.obs_adapter.fit(
                self.policy_manager.override_dof_pos
            )

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)

        if self.return_blend_active:
            pd_target = self.blend_return_target(
                self.return_blend_start,
                pd_target,
                self.return_blend_step,
                self.return_blend_duration,
            )
            self.return_blend_step += 1
            if self.return_blend_step >= self.return_blend_duration:
                self.return_blend_active = False
                self.control_state = ControlState.LOCO

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)
