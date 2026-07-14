# RoboJuDo G1 Policy Integration

This repository is a customized RoboJuDo project for Unitree G1 sim2sim experiments. It integrates one locomotion policy, multiple learned action policies, keyboard control, and an AMP-based recovery/stabilization state into RoboJuDo's modular deployment framework.

The current default workflow is MuJoCo simulation first. Real-robot deployment still requires Unitree SDK installation, network setup, and hardware safety validation.

## Current Features

- Unitree G1 MuJoCo sim2sim pipeline.
- Keyboard-controlled locomotion.
- Runtime switching between locomotion, learned actions, and recovery.
- MJLab locomotion policy loaded from `assets/models/g1/mjlab/locomotion_v3`.
- Seven MJLab action policies loaded from `assets/models/g1/mjlab/actions`.
- AMP policy used as manual recovery and post-action stabilization.
- Smooth return blend from action/recovery back to locomotion.
- Tests covering MJLab velocity policy, tracking policy, and AMP recovery pipeline behavior.

## Repository Layout

```text
RoboJuDo/
|-- assets/models/g1/mjlab/
|   |-- locomotion_v1/          # older locomotion policy retained for rollback
|   |-- locomotion_v2/          # older locomotion policy retained for rollback
|   |-- locomotion_v3/          # current locomotion policy
|   `-- actions/                # integrated action policies
|-- robojudo/
|   |-- config/g1/              # G1 task, policy, env, and controller configs
|   |-- controller/             # keyboard/joystick/unitree controller modules
|   |-- environment/            # MuJoCo and real robot environment wrappers
|   |-- pipeline/               # policy switching state machine
|   `-- policy/                 # policy runtime implementations
|-- scripts/run_pipeline.py     # main entry point
|-- tests/                      # focused regression tests
`-- requirements.txt
```

## Integrated Policies

### Locomotion

The active locomotion policy is:

```text
assets/models/g1/mjlab/locomotion_v3/policy.onnx
assets/models/g1/mjlab/locomotion_v3/params/env.yaml
assets/models/g1/mjlab/locomotion_v3/params/agent.yaml
```

It is selected by `G1MjlabVelocityPolicyCfg.policy_name = "locomotion_v3"`.

### Actions

Each action directory should contain:

```text
policy.onnx
motion.npz
params/env.yaml
params/agent.yaml
```

Current action order:

| Key | Policy name | Directory |
| --- | --- | --- |
| `1` | `right_overhand` | `assets/models/g1/mjlab/actions/right_overhand/` |
| `2` | `back_kick` | `assets/models/g1/mjlab/actions/back_kick/` |
| `3` | `rear_straight_punch` | `assets/models/g1/mjlab/actions/rear_straight_punch/` |
| `4` | `left_jab` | `assets/models/g1/mjlab/actions/left_jab/` |
| `5` | `right_cross` | `assets/models/g1/mjlab/actions/right_cross/` |
| `6` | `left_front_kick` | `assets/models/g1/mjlab/actions/left_front_kick/` |
| `7` | `spin_kick` | `assets/models/g1/mjlab/actions/spin_kick/` |

The order is defined in `robojudo/config/g1/g1_custom_cfg.py` by `mimic_policies`.

## State Machine

The custom pipeline is `MjlabLocoActionPipeline` in:

```text
robojudo/pipeline/mjlab_loco_action_pipeline.py
```

Current states:

| State | Meaning |
| --- | --- |
| `LOCO` | normal locomotion policy |
| `ACTION` | selected learned action policy |
| `RETURN` | blended transition back to locomotion |
| `STABILIZE` | AMP policy used after action 3 |
| `RECOVERY` | manual AMP recovery mode |

Current transition behavior:

- Keys `1`, `2`, `4`, `5`, `6`, `7`: `ACTION -> RETURN -> LOCO`.
- Key `3`: `ACTION -> STABILIZE -> RETURN -> LOCO`.
- Key `9`: manually enters `RECOVERY`.
- Key `0`: exits recovery/stabilization only after the robot is stably upright.
- Action switching is accepted only from `LOCO`, not during `ACTION` or `RETURN`.

The config `g1_mjlab_loco_right_overhand_post_action_only` disables the in-action disturbance guard and keeps only the post-action stabilization behavior for action 3.

## Keyboard Control

Run the simulation window first, then focus the terminal that captures keyboard input.

### Locomotion Commands

| Key | Command |
| --- | --- |
| `W` | forward velocity `+0.5 m/s` |
| `S` | backward velocity `-0.3 m/s` |
| `A` | left lateral velocity `+0.2 m/s` |
| `D` | right lateral velocity `-0.2 m/s` |
| `Q` | yaw left `+0.3 rad/s` |
| `E` | yaw right `-0.3 rad/s` |
| `Space` | zero velocity command |

Velocity commands latch when pressed. Releasing the key does not automatically stop the robot. Press `Space` to stop walking.

### State/Action Commands

Action and mode commands are event-triggered on key release.

| Key | Command |
| --- | --- |
| `1` | run `right_overhand` |
| `2` | run `back_kick` |
| `3` | run `rear_straight_punch`, then stabilize |
| `4` | run `left_jab` |
| `5` | run `right_cross` |
| `6` | run `left_front_kick` |
| `7` | run `spin_kick` |
| `9` | manual AMP recovery |
| `0` | return to locomotion |
| `` ` `` | reset/reborn simulation |
| `Esc` | shutdown command |

## Quick Start: MuJoCo Sim2Sim

From the project root:

```bash
cd /root/gpufree-data/RoboJuDo
conda activate robojudo
python scripts/run_pipeline.py -c g1_mjlab_loco_right_overhand_post_action_only
```

Recommended basic test flow:

1. Start the command above.
2. Press `Space` to make sure velocity is zero.
3. Press `W`, `A`, `S`, `D`, `Q`, `E` to verify locomotion control.
4. Press `Space` again to stop.
5. Press `1` through `7` to test action policies one by one.
6. If the robot falls, press `9` for AMP recovery, wait until it is upright and stable, then press `0` to return to locomotion.

## Development Setup

Create and activate the environment:

```bash
conda create -n robojudo python=3.11 -y
conda activate robojudo
pip install -e .
```

For CPU-only PyTorch installation:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Optional modules are configured through:

```text
submodule_cfg.yaml
```

Install selected optional modules with:

```bash
python submodule_install.py
```

## Tests

Run the focused tests for the customized MJLab integration:

```bash
pytest tests/test_mjlab_velocity_policy.py
pytest tests/test_mjlab_tracking_policy.py
pytest tests/test_amp_recovery_pipeline.py
```

Known optional dependency failures in the full test suite:

- `phc` and `redis` are missing for some controller imports.
- `unitree_sdk2py` and `unitree_cpp` are missing for real robot environment imports.

These are not required for the current MuJoCo sim2sim workflow, but they must be handled before real-robot deployment.

## Replacing Policies

### Replace locomotion

1. Put the new locomotion files under a new directory, for example:

```text
assets/models/g1/mjlab/locomotion_v4/
|-- policy.onnx
`-- params/
    |-- env.yaml
    `-- agent.yaml
```

2. Edit:

```text
robojudo/config/g1/policy/g1_mjlab_velocity_policy_cfg.py
```

3. Change:

```python
policy_name: str = "locomotion_v4"
```

4. Run the velocity policy tests.

### Add an action

1. Add the action directory:

```text
assets/models/g1/mjlab/actions/new_action/
|-- policy.onnx
|-- motion.npz
`-- params/
    |-- env.yaml
    `-- agent.yaml
```

2. Edit:

```text
robojudo/config/g1/g1_custom_cfg.py
```

3. Append the policy to `mimic_policies`:

```python
G1MjlabTrackingPolicyCfg(policy_name="new_action"),
```

4. Add a keyboard trigger pointing to the corresponding mimic index.

5. Run the tracking and pipeline tests.

## Real Robot Deployment Notes

The current custom config uses `G1MujocoEnvCfg`, so it is a simulation config. Do not run it directly on the real robot as-is.

Before real deployment, verify at least the following:

- `unitree_sdk2py` or `unitree_cpp` is installed and importable.
- The controller runs on the robot PC or an onboard/onsite machine connected to the Unitree DDS network.
- The real G1 has the same controlled DoF layout as the policy, including waist and wrist joints.
- Motor order, PD gains, torque limits, and action scaling match the real hardware.
- Emergency stop and damping mode are independently available.
- Recovery/stabilization thresholds are recalibrated using real IMU and odometry data.
- First tests are done with the robot safely suspended or supported.

## GitHub Upload Notes

This repository contains binary policy artifacts such as `.onnx`, `.npz`, and possibly `.pt` files. If GitHub rejects large files, use Git LFS:

```bash
git lfs install
git lfs track "*.onnx"
git lfs track "*.npz"
git lfs track "*.pt"
git add .gitattributes
```

Then commit and push normally.

## License and Upstream

This project is based on RoboJuDo by HansZ8. Keep the upstream license and attribution when publishing modified versions.

Upstream repository:

```text
https://github.com/HansZ8/RoboJuDo
```
