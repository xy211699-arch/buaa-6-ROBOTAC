import os
from collections import deque

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.utils.util_func import quat_rotate_inverse_np


@policy_registry.register
class AmpRecoveryPolicy(Policy):
    """Adapter for the wbc_fsm 29-DoF AMP policy used as recovery."""

    frame_observation_size = 96
    observation_size = 384

    def __init__(self, cfg_policy, device="cpu"):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        self.session = ort.InferenceSession(
            cfg_policy.policy_file, providers=["CPUExecutionProvider"]
        )
        self._validate_model_contract()

        super().__init__(cfg_policy=cfg_policy, device="cpu")

        self.action_scales = np.asarray(
            cfg_policy.action_scales, dtype=np.float32
        )
        if self.action_scales.shape != (29,):
            raise ValueError(
                f"Expected 29 action scales, got {self.action_scales.shape}"
            )

        self.observation_clip = cfg_policy.observation_clip
        self._history = deque(maxlen=self.history_length)
        self.reset()

    def _validate_model_contract(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        input_valid = (
            len(inputs) == 1
            and inputs[0].name == "obs"
            and len(inputs[0].shape) == 2
            and inputs[0].shape[1] == self.observation_size
        )
        if not input_valid:
            raise ValueError(
                "Expected ONNX input obs with shape [batch, 384], got "
                f"{[(item.name, item.shape) for item in inputs]}"
            )

        output_valid = (
            len(outputs) >= 1
            and outputs[0].name == "actions"
            and len(outputs[0].shape) == 2
            and outputs[0].shape[1] == 29
        )
        if not output_valid:
            raise ValueError(
                "Expected ONNX output actions with shape [batch, 29], got "
                f"{[(item.name, item.shape) for item in outputs]}"
            )

    def reset(self):
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._history.clear()

    def post_step_callback(self, commands=None):
        return

    def _current_frame(self, env_data):
        base_quat = np.asarray(env_data.base_quat, dtype=np.float32)
        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)

        if base_quat.shape != (4,) or not np.isfinite(base_quat).all():
            raise ValueError("base_quat must be a finite xyzw quaternion")
        if base_ang_vel.shape != (3,) or not np.isfinite(base_ang_vel).all():
            raise ValueError("base_ang_vel must have shape (3,) and be finite")
        if dof_pos.shape != (29,) or dof_vel.shape != (29,):
            raise ValueError("Expected 29 joint positions and velocities")
        if not np.isfinite(dof_pos).all() or not np.isfinite(dof_vel).all():
            raise ValueError("Joint state contains non-finite values")

        projected_gravity = quat_rotate_inverse_np(
            base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        ).astype(np.float32)
        frame = np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                np.zeros(3, dtype=np.float32),
                dof_pos - self.default_dof_pos,
                dof_vel,
                self.last_action,
            ]
        ).astype(np.float32, copy=False)
        if frame.shape != (self.frame_observation_size,):
            raise ValueError(
                "Expected AMP frame observation shape "
                f"({self.frame_observation_size},), got {frame.shape}"
            )
        return np.clip(frame, -self.observation_clip, self.observation_clip)

    def get_observation(self, env_data, ctrl_data):
        frame = self._current_frame(env_data)
        if not self._history:
            for _ in range(self.history_length):
                self._history.append(frame.copy())
        else:
            self._history.append(frame.copy())

        obs = np.concatenate(tuple(self._history)).astype(
            np.float32, copy=False
        )
        if obs.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape ({self.observation_size},), "
                f"got {obs.shape}"
            )
        return obs, {"CALLBACK": [], "commands": np.zeros(3, dtype=np.float32)}

    def get_action(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape ({self.observation_size},), "
                f"got {obs.shape}"
            )
        if not np.isfinite(obs).all():
            raise ValueError("AMP recovery observation contains non-finite values")

        raw_action = np.asarray(
            self.session.run(["actions"], {"obs": obs[None, :]})[0]
        ).squeeze()
        if raw_action.shape != (self.num_actions,):
            raise ValueError(
                f"Expected action shape ({self.num_actions},), got {raw_action.shape}"
            )
        if not np.isfinite(raw_action).all():
            raise ValueError("AMP recovery policy returned non-finite actions")

        if self.action_clip is not None:
            raw_action = np.clip(raw_action, -self.action_clip, self.action_clip)
        self.last_action = raw_action.astype(np.float32, copy=True)
        return self.last_action * self.action_scales