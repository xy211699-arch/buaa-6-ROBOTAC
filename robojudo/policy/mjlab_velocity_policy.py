import os

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.util_func import quat_rotate_inverse_np


@policy_registry.register
class MjlabVelocityPolicy(Policy):
    """Adapter for a 29-DoF MjLab velocity policy exported to ONNX."""

    observation_size = 98

    def __init__(self, cfg_policy, device="cpu"):
        policy_file = cfg_policy.policy_file
        if not os.path.isfile(policy_file):
            raise FileNotFoundError(f"Model file not found at {policy_file}")

        self.session = ort.InferenceSession(
            policy_file, providers=["CPUExecutionProvider"]
        )
        self._validate_model_contract()

        metadata = self.session.get_modelmeta().custom_metadata_map
        joint_names = self._parse_names(metadata, "joint_names")
        default_pos = self._parse_floats(metadata, "default_joint_pos")
        stiffness = self._parse_floats(metadata, "joint_stiffness")
        damping = self._parse_floats(metadata, "joint_damping")
        self.action_scales = np.asarray(
            self._parse_floats(metadata, "action_scale"), dtype=np.float32
        )

        expected_length = len(joint_names)
        for name, values in (
            ("default_joint_pos", default_pos),
            ("joint_stiffness", stiffness),
            ("joint_damping", damping),
            ("action_scale", self.action_scales),
        ):
            if len(values) != expected_length:
                raise ValueError(
                    f"Metadata {name} has {len(values)} values; "
                    f"expected {expected_length}"
                )

        dof_cfg = DoFConfig(
            joint_names=joint_names,
            default_pos=default_pos,
            stiffness=stiffness,
            damping=damping,
        )
        cfg_policy_updated = cfg_policy.model_copy()
        cfg_policy_updated.obs_dof = dof_cfg
        cfg_policy_updated.action_dof = dof_cfg
        super().__init__(cfg_policy=cfg_policy_updated, device="cpu")

        self.gait_period = cfg_policy.gait_period
        self._command_map = {
            "w": np.array([cfg_policy.command_forward, 0.0, 0.0]),
            "s": np.array([-cfg_policy.command_backward, 0.0, 0.0]),
            "a": np.array([0.0, cfg_policy.command_lateral, 0.0]),
            "d": np.array([0.0, -cfg_policy.command_lateral, 0.0]),
            "q": np.array([0.0, 0.0, cfg_policy.command_yaw]),
            "e": np.array([0.0, 0.0, -cfg_policy.command_yaw]),
        }
        self.reset()

    def _validate_model_contract(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape != [1, 98]:
            raise ValueError(
                "Expected ONNX input obs with shape [1, 98], got "
                f"{[(item.name, item.shape) for item in inputs]}"
            )
        if not outputs or outputs[0].name != "actions" or outputs[0].shape != [1, 29]:
            raise ValueError(
                "Expected ONNX output actions with shape [1, 29], got "
                f"{[(item.name, item.shape) for item in outputs]}"
            )

    @staticmethod
    def _metadata_value(metadata, key):
        value = metadata.get(key)
        if not value:
            raise ValueError(f"ONNX metadata is missing required key: {key}")
        return value

    @classmethod
    def _parse_names(cls, metadata, key):
        return [value.strip() for value in cls._metadata_value(metadata, key).split(",")]

    @classmethod
    def _parse_floats(cls, metadata, key):
        return [float(value) for value in cls._metadata_value(metadata, key).split(",")]

    def reset(self):
        self.phase = 0.0
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def stop(self):
        self.command.fill(0.0)

    def post_step_callback(self, commands=None):
        self.phase = (self.phase + self.dt / self.gait_period) % 1.0

    def _update_command(self, ctrl_data):
        keyboard_data = ctrl_data.get("KeyboardCtrl", {})
        for event in keyboard_data.get("keyboard_event", []):
            if event.get("type") != "keyboard" or not event.get("pressed"):
                continue
            key = event.get("name")
            if key in self._command_map:
                self.command = self._command_map[key].astype(np.float32, copy=True)
            elif key in ("Key.space", " "):
                self.command.fill(0.0)

    def get_observation(self, env_data, ctrl_data):
        self._update_command(ctrl_data)

        base_quat = np.asarray(env_data.base_quat, dtype=np.float32)
        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        projected_gravity = np.asarray(
            quat_rotate_inverse_np(base_quat, np.array([0.0, 0.0, -1.0])),
            dtype=np.float32,
        )

        if np.linalg.norm(self.command) < 0.1:
            phase = np.zeros(2, dtype=np.float32)
        else:
            angle = 2.0 * np.pi * self.phase
            phase = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

        obs = np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                self.command,
                phase,
                dof_pos - self.default_dof_pos,
                dof_vel,
                self.last_action,
            ]
        ).astype(np.float32, copy=False)
        if obs.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape ({self.observation_size},), got {obs.shape}"
            )

        return obs, {"commands": self.command.copy()}

    def get_action(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape ({self.observation_size},), got {obs.shape}"
            )
        raw_action = np.asarray(
            self.session.run(["actions"], {"obs": obs[None, :]})[0]
        ).squeeze()
        if raw_action.shape != (self.num_actions,):
            raise ValueError(
                f"Expected action shape ({self.num_actions},), got {raw_action.shape}"
            )
        if not np.isfinite(raw_action).all():
            raise ValueError("Policy returned non-finite actions")

        self.last_action = raw_action.astype(np.float32, copy=True)
        return self.last_action * self.action_scales

    def get_init_dof_pos(self):
        return self.default_pos.copy()
