from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_DOF_POS = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]

STIFFNESS = [
    40.17923863450712, 99.09842777666111, 40.17923863450712,
    99.09842777666111, 28.50124619574858, 28.50124619574858,
    40.17923863450712, 99.09842777666111, 40.17923863450712,
    99.09842777666111, 28.50124619574858, 28.50124619574858,
    40.17923863450712, 28.50124619574858, 28.50124619574858,
    14.25062309787429, 14.25062309787429, 14.25062309787429,
    14.25062309787429, 14.25062309787429, 16.77832748089279,
    16.77832748089279, 14.25062309787429, 14.25062309787429,
    14.25062309787429, 14.25062309787429, 14.25062309787429,
    16.77832748089279, 16.77832748089279,
]

DAMPING = [
    2.557889775413375, 6.308801853496639, 2.557889775413375,
    6.308801853496639, 1.814445686584846, 1.814445686584846,
    2.557889775413375, 6.308801853496639, 2.557889775413375,
    6.308801853496639, 1.814445686584846, 1.814445686584846,
    2.557889775413375, 1.814445686584846, 1.814445686584846,
    0.907222843292423, 0.907222843292423, 0.907222843292423,
    0.907222843292423, 0.907222843292423, 1.06814150219,
    1.06814150219, 0.907222843292423, 0.907222843292423,
    0.907222843292423, 0.907222843292423, 0.907222843292423,
    1.06814150219, 1.06814150219,
]

ACTION_SCALES = [
    0.54754646, 0.35066147, 0.54754646, 0.35066147, 0.43857731,
    0.43857731, 0.54754646, 0.35066147, 0.54754646, 0.35066147,
    0.43857731, 0.43857731, 0.54754646, 0.43857731, 0.43857731,
    0.43857731, 0.43857731, 0.43857731, 0.43857731, 0.43857731,
    0.07450087, 0.07450087, 0.43857731, 0.43857731, 0.43857731,
    0.43857731, 0.43857731, 0.07450087, 0.07450087,
]


class G1MjlabTrackingPolicyCfg(PolicyCfg):
    policy_type: str = "MjlabTrackingPolicy"
    robot: str = "g1"
    policy_name: str = "right_overhand"
    disable_autoload: bool = True

    model_path: str | None = None
    motion_path: str | None = None

    @property
    def policy_file(self) -> str:
        if self.model_path is not None:
            return self.model_path
        return (
            ASSETS_DIR
            / f"models/{self.robot}/mjlab/actions/{self.policy_name}/policy.onnx"
        ).as_posix()

    @property
    def motion_file(self) -> str:
        if self.motion_path is not None:
            return self.motion_path
        return (
            ASSETS_DIR
            / f"models/{self.robot}/mjlab/actions/{self.policy_name}/motion.npz"
        ).as_posix()

    freq: int = 50
    anchor_body_index: int = 15
    gravity_error_threshold: float = 0.6
    observation_clip: float = 100.0
    action_scales: list[float] = ACTION_SCALES

    action_scale: float = 1.0
    action_clip: float | None = 100.0
    action_beta: float = 1.0

    obs_dof: DoFConfig = DoFConfig(
        joint_names=JOINT_NAMES,
        default_pos=DEFAULT_DOF_POS,
        stiffness=STIFFNESS,
        damping=DAMPING,
    )
    action_dof: DoFConfig = obs_dof
