from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from .g1_mjlab_tracking_policy_cfg import DEFAULT_DOF_POS, JOINT_NAMES


_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_5010_16 = 0.0021812

_EFFECTIVE_ARMATURES = [
    _ARMATURE_7520_22, _ARMATURE_7520_22, _ARMATURE_7520_14,
    _ARMATURE_7520_22, 2.0 * _ARMATURE_5020, 2.0 * _ARMATURE_5020,
    _ARMATURE_7520_22, _ARMATURE_7520_22, _ARMATURE_7520_14,
    _ARMATURE_7520_22, 2.0 * _ARMATURE_5020, 2.0 * _ARMATURE_5020,
    _ARMATURE_7520_14, 2.0 * _ARMATURE_5020, 2.0 * _ARMATURE_5020,
    _ARMATURE_5020, _ARMATURE_5020, _ARMATURE_5020, _ARMATURE_5020,
    _ARMATURE_5020, _ARMATURE_5010_16, _ARMATURE_5010_16,
    _ARMATURE_5020, _ARMATURE_5020, _ARMATURE_5020, _ARMATURE_5020,
    _ARMATURE_5020, _ARMATURE_5010_16, _ARMATURE_5010_16,
]

TORQUE_LIMITS = [
    139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 10.0, 10.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 10.0, 10.0,
]
STIFFNESS = [
    armature * _NATURAL_FREQ * _NATURAL_FREQ
    for armature in _EFFECTIVE_ARMATURES
]
DAMPING = [
    2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
    for armature in _EFFECTIVE_ARMATURES
]
ACTION_SCALES = [
    0.25 * torque_limit / stiffness
    for torque_limit, stiffness in zip(TORQUE_LIMITS, STIFFNESS, strict=True)
]


class G1AmpRecoveryPolicyCfg(PolicyCfg):
    policy_type: str = "AmpRecoveryPolicy"
    robot: str = "g1"
    policy_name: str = "amp_recovery"
    disable_autoload: bool = True

    model_path: str | None = None

    @property
    def policy_file(self) -> str:
        if self.model_path is not None:
            return self.model_path
        return (
            ASSETS_DIR
            / f"models/{self.robot}/wbc_fsm/{self.policy_name}/policy.onnx"
        ).as_posix()

    freq: int = 50
    observation_clip: float = 100.0
    action_scales: list[float] = ACTION_SCALES

    action_scale: float = 1.0
    action_clip: float | None = 100.0
    action_beta: float = 1.0

    history_length: int = 4

    @property
    def history_obs_size(self) -> int:
        return 96

    obs_dof: DoFConfig = DoFConfig(
        joint_names=JOINT_NAMES,
        default_pos=DEFAULT_DOF_POS,
        stiffness=STIFFNESS,
        damping=DAMPING,
        torque_limits=TORQUE_LIMITS,
    )
    action_dof: DoFConfig = obs_dof