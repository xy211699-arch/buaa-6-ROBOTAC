from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1MjlabVelocityPolicyCfg(PolicyCfg):
    policy_type: str = "MjlabVelocityPolicy"
    robot: str = "g1"
    policy_name: str = "locomotion_v3"
    disable_autoload: bool = True

    model_path: str | None = None

    @property
    def policy_file(self) -> str:
        if self.model_path is not None:
            return self.model_path
        return (
            ASSETS_DIR
            / f"models/{self.robot}/mjlab/{self.policy_name}/policy.onnx"
        ).as_posix()

    freq: int = 50
    gait_period: float = 0.6

    command_forward: float = 0.5
    command_backward: float = 0.3
    command_lateral: float = 0.2
    command_yaw: float = 0.3

    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    # Replaced from ONNX metadata before the base policy is initialized.
    obs_dof: DoFConfig = DoFConfig(
        joint_names=["placeholder"], default_pos=[0.0]
    )
    action_dof: DoFConfig = obs_dof
