import os

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as sRot

from robojudo.policy import Policy, policy_registry


@policy_registry.register
class MjlabTrackingPolicy(Policy):
    """Adapter for a 29-DoF MjLab motion-tracking policy."""

    observation_size = 154

    def __init__(self, cfg_policy, device="cpu"):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")
        if not os.path.isfile(cfg_policy.motion_file):
            raise FileNotFoundError(f"Motion file not found at {cfg_policy.motion_file}")

        self.session = ort.InferenceSession(
            cfg_policy.policy_file, providers=["CPUExecutionProvider"]
        )
        self._validate_model_contract()
        self._load_motion(cfg_policy.motion_file, cfg_policy.anchor_body_index)

        self.action_scales = np.asarray(
            cfg_policy.action_scales, dtype=np.float32
        )
        if self.action_scales.shape != (29,):
            raise ValueError(
                f"Expected 29 action scales, got {self.action_scales.shape}"
            )

        super().__init__(cfg_policy=cfg_policy, device="cpu")
        self.anchor_body_index = cfg_policy.anchor_body_index
        self.gravity_error_threshold = cfg_policy.gravity_error_threshold
        self.observation_clip = cfg_policy.observation_clip
        self.reset()

    def _validate_model_contract(self):
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape != [1, 154]:
            raise ValueError(
                "Expected ONNX input obs with shape [1, 154], got "
                f"{[(item.name, item.shape) for item in inputs]}"
            )
        if not outputs or outputs[0].name != "actions" or outputs[0].shape != [1, 29]:
            raise ValueError(
                "Expected ONNX output actions with shape [1, 29], got "
                f"{[(item.name, item.shape) for item in outputs]}"
            )

    def _load_motion(self, motion_file, anchor_body_index):
        with np.load(motion_file, allow_pickle=False) as motion:
            required = {"fps", "joint_pos", "joint_vel", "body_quat_w"}
            missing = required.difference(motion.files)
            if missing:
                raise ValueError(f"Motion file is missing fields: {sorted(missing)}")
            self.reference_joint_pos = np.asarray(
                motion["joint_pos"], dtype=np.float32
            ).copy()
            self.reference_joint_vel = np.asarray(
                motion["joint_vel"], dtype=np.float32
            ).copy()
            self.reference_body_quat_wxyz = np.asarray(
                motion["body_quat_w"], dtype=np.float32
            ).copy()
            fps = np.asarray(motion["fps"], dtype=np.float64).reshape(-1)

        if self.reference_joint_pos.ndim != 2 or self.reference_joint_pos.shape[1] != 29:
            raise ValueError(
                "Motion joint_pos must have shape [frames, 29], got "
                f"{self.reference_joint_pos.shape}"
            )
        if self.reference_joint_vel.shape != self.reference_joint_pos.shape:
            raise ValueError("Motion joint_vel shape must match joint_pos")
        if (
            self.reference_body_quat_wxyz.ndim != 3
            or self.reference_body_quat_wxyz.shape[0]
            != self.reference_joint_pos.shape[0]
            or self.reference_body_quat_wxyz.shape[2] != 4
            or not 0 <= anchor_body_index < self.reference_body_quat_wxyz.shape[1]
        ):
            raise ValueError("Motion body_quat_w has an invalid shape or anchor index")
        if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
            raise ValueError("Motion fps must contain one positive finite value")

        arrays = (
            self.reference_joint_pos,
            self.reference_joint_vel,
            self.reference_body_quat_wxyz,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Motion file contains non-finite values")

        self.motion_frame_count = self.reference_joint_pos.shape[0]
        self.motion_fps = float(fps[0])

    @staticmethod
    def _wxyz_to_xyzw(quat):
        quat = np.asarray(quat, dtype=np.float64)
        return quat[[1, 2, 3, 0]]

    @staticmethod
    def _yaw_rotation(quat_xyzw):
        matrix = sRot.from_quat(quat_xyzw).as_matrix()
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        return sRot.from_euler("z", yaw)

    def _aligned_reference_rotation(self, base_quat):
        reference_quat = self._wxyz_to_xyzw(
            self.reference_body_quat_wxyz[
                self.motion_frame, self.anchor_body_index
            ]
        )
        reference_rotation = sRot.from_quat(reference_quat)
        if self.reference_yaw_alignment is None:
            reference_root_start = self._wxyz_to_xyzw(
                self.reference_body_quat_wxyz[0, 0]
            )
            self.reference_yaw_alignment = (
                self._yaw_rotation(base_quat)
                * self._yaw_rotation(reference_root_start).inv()
            )
        return self.reference_yaw_alignment * reference_rotation

    def reset(self):
        self.motion_time = 0.0
        self.motion_frame = 0
        self.reference_yaw_alignment = None
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._terminated = False
        self._completion_emitted = False

    def reset_alignment(self):
        self.reference_yaw_alignment = None

    def post_step_callback(self, commands=None):
        if self._terminated or self.motion_frame >= self.motion_frame_count - 1:
            return
        self.motion_time += self.dt
        self.motion_frame = min(
            int(np.floor(self.motion_time * self.motion_fps)),
            self.motion_frame_count - 1,
        )

    def get_observation(self, env_data, ctrl_data):
        base_quat = np.asarray(env_data.base_quat, dtype=np.float64)
        torso_quat = np.asarray(env_data.torso_quat, dtype=np.float64)
        if base_quat.shape != (4,) or not np.isfinite(base_quat).all():
            raise ValueError("base_quat must be a finite xyzw quaternion")
        if torso_quat.shape != (4,) or not np.isfinite(torso_quat).all():
            raise ValueError("torso_quat must be a finite xyzw quaternion")

        aligned_reference = self._aligned_reference_rotation(base_quat)
        torso_rotation = sRot.from_quat(torso_quat)
        target_rotation = torso_rotation.inv() * aligned_reference
        target_orientation_6d = target_rotation.as_matrix()[:, :2].reshape(-1)

        gravity = np.array([0.0, 0.0, -1.0])
        reference_gravity = aligned_reference.inv().apply(gravity)
        measured_gravity = torso_rotation.inv().apply(gravity)
        gravity_error = float(abs(reference_gravity[2] - measured_gravity[2]))
        if gravity_error > self.gravity_error_threshold:
            self._terminated = True

        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
        if dof_pos.shape != (29,) or dof_vel.shape != (29,):
            raise ValueError("Expected 29 joint positions and velocities")
        if base_ang_vel.shape != (3,):
            raise ValueError("Expected base_ang_vel with shape (3,)")

        reference_observation = np.concatenate(
            [
                self.reference_joint_pos[self.motion_frame],
                self.reference_joint_vel[self.motion_frame],
                target_orientation_6d,
            ]
        )
        robot_state_observation = np.concatenate(
            [
                base_ang_vel,
                dof_pos - self.default_dof_pos,
                dof_vel,
                self.last_action,
            ]
        )
        obs = np.concatenate(
            [reference_observation, robot_state_observation]
        ).astype(np.float32, copy=False)
        if obs.shape != (self.observation_size,):
            raise ValueError(
                f"Expected observation shape ({self.observation_size},), got {obs.shape}"
            )
        if not np.isfinite(obs).all():
            raise ValueError("Tracking observation contains non-finite values")
        obs = np.clip(obs, -self.observation_clip, self.observation_clip)

        callbacks = []
        motion_finished = self.motion_frame >= self.motion_frame_count - 1
        if (motion_finished or self._terminated) and not self._completion_emitted:
            callbacks.append("[MOTION_DONE]")
            self._completion_emitted = True

        return obs, {
            "CALLBACK": callbacks,
            "motion_frame": self.motion_frame,
            "gravity_error": gravity_error,
        }

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

        if self.action_clip is not None:
            raw_action = np.clip(raw_action, -self.action_clip, self.action_clip)
        self.last_action = raw_action.astype(np.float32, copy=True)
        return self.last_action * self.action_scales

    def get_init_dof_pos(self):
        return self.reference_joint_pos[0].copy()
