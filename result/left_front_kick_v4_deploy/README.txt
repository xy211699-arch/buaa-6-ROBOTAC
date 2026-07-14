ROBOTAC / Unitree G1 Deployment Package

Skill: left_front_kick_v4
Training task: Unitree-G1-Tracking
Main checkpoint: model_29000.pt
Deployment policy: policy.onnx

Directory layout:
policy.onnx
model_29000.pt
motion/
├── motion.npz
├── motion.pkl
└── motion.csv
params/
├── agent.yaml
└── env.yaml
unitree_rl_mjlab.diff
README.txt

Usage notes:
1. policy.onnx is the primary policy file for deployment-side inference.
2. model_29000.pt is kept for debugging, continuing training, or re-exporting ONNX.
3. params/env.yaml and params/agent.yaml define the training task, observation/action setup, and policy config.
4. motion/motion.npz is the expert motion used for training/replay.
5. motion/motion.csv is the source motion trajectory.
6. motion/motion.pkl was derived from motion.csv and contains root_pos, root_rot, and dof_pos only.
7. unitree_rl_mjlab.diff is currently a placeholder because the original diff file was not included in the uploaded files. Replace it with the real server-side diff for exact code reproducibility.
