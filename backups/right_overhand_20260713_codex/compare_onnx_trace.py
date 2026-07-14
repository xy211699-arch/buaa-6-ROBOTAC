"""Compare an exported ONNX policy with a captured MjLab PT trace."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


RESULT_DIR = Path("/root/gpufree-data/unitree_rl_mjlab/result/motions/right_overhand")
TRACE_FILE = Path(
    "/root/gpufree-data/RoboJuDo/backups/right_overhand_20260713_codex/"
    "mjlab_pt_trace.npz"
)


def main() -> None:
    trace = np.load(TRACE_FILE)
    observations = trace["observations"]
    expected_actions = trace["actions"]
    session = ort.InferenceSession(
        str(RESULT_DIR / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    actual_actions = np.concatenate(
        [
            session.run(["actions"], {"obs": observation[None, :]})[0]
            for observation in observations
        ],
        axis=0,
    )
    absolute_error = np.abs(expected_actions - actual_actions)
    print(f"samples={len(observations)}")
    print(f"max_abs_error={absolute_error.max():.9f}")
    print(f"mean_abs_error={absolute_error.mean():.9f}")
    print(f"pt_first_norm={np.linalg.norm(expected_actions[0]):.9f}")
    print(f"onnx_first_norm={np.linalg.norm(actual_actions[0]):.9f}")


if __name__ == "__main__":
    main()
