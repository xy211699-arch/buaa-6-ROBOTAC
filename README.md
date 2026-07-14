# BUAA-6-ROBOTAC项目

本项目基于 [RoboJuDo](https://github.com/HansZ8/RoboJuDo) 进行定制开发，用于 Unitree G1 人形机器人的策略集成与 MuJoCo sim2sim 仿真验证。

当前项目集成了：

- 1 个基于 MJLab 训练的行走策略；
- 7 个基于 MJLab 训练的动作策略；
- 键盘控制模块；
- 策略运行时切换状态机；
- 基于 AMP 策略的恢复与稳定控制；
- MuJoCo 环境下的 sim2sim 测试流程。

当前默认工作流程为 **MuJoCo 仿真优先**。

> [!WARNING]
> 当前默认配置使用 MuJoCo 仿真环境，不能直接用于真实机器人。
>
> 在部署到真实 Unitree G1 之前，必须完成 Unitree SDK、DDS 网络、关节映射、控制增益、限位保护和急停机制等配置，并在机器人悬挂或有安全支撑的条件下进行首次测试。

---

## 当前功能

- 支持 Unitree G1 的 MuJoCo sim2sim 仿真；
- 支持通过键盘控制机器人行走；
- 支持在运行过程中切换行走策略、动作策略和恢复策略；
- 加载位于 `assets/models/g1/mjlab/locomotion_v3` 的 MJLab 行走策略；
- 加载位于 `assets/models/g1/mjlab/actions` 的 7 个 MJLab 动作策略；
- 使用 AMP 策略进行手动恢复；
- 使用 AMP 策略进行指定动作结束后的姿态稳定；
- 动作策略或恢复策略结束后，可平滑过渡回行走策略；
- 提供针对行走策略、动作跟踪策略和 AMP 恢复流程的回归测试。

---

## 项目目录结构

```text
RoboJuDo/
├── assets/models/g1/mjlab/
│   ├── locomotion_v1/          # 旧版行走策略，用于回滚
│   ├── locomotion_v2/          # 旧版行走策略，用于回滚
│   ├── locomotion_v3/          # 当前使用的行走策略
│   └── actions/                # 已集成的动作策略
├── robojudo/
│   ├── config/g1/              # G1 任务、策略、环境和控制器配置
│   ├── controller/             # 键盘、手柄和 Unitree 控制器模块
│   ├── environment/            # MuJoCo 与真实机器人环境封装
│   ├── pipeline/               # 策略切换状态机
│   └── policy/                 # 策略运行时实现
├── scripts/
│   └── run_pipeline.py         # 主程序入口
├── tests/                      # 定制功能的回归测试
├── requirements.txt
└── submodule_cfg.yaml          # 可选子模块配置
```

---

## 已集成策略

### 行走策略

当前使用的行走策略文件为：

```text
assets/models/g1/mjlab/locomotion_v3/policy.onnx
assets/models/g1/mjlab/locomotion_v3/params/env.yaml
assets/models/g1/mjlab/locomotion_v3/params/agent.yaml
```

### 动作策略

每个动作策略目录应包含以下文件：

```text
policy.onnx
motion.npz
params/env.yaml
params/agent.yaml
```

当前动作策略及其键盘映射如下：

| 按键 | 策略名称 | 策略目录 |
| --- | --- | --- |
| `1` | `right_overhand` | `assets/models/g1/mjlab/actions/right_overhand/` |
| `2` | `back_kick` | `assets/models/g1/mjlab/actions/back_kick/` |
| `3` | `rear_straight_punch` | `assets/models/g1/mjlab/actions/rear_straight_punch/` |
| `4` | `left_jab` | `assets/models/g1/mjlab/actions/left_jab/` |
| `5` | `right_cross` | `assets/models/g1/mjlab/actions/right_cross/` |
| `6` | `left_front_kick` | `assets/models/g1/mjlab/actions/left_front_kick/` |
| `7` | `spin_kick` | `assets/models/g1/mjlab/actions/spin_kick/` |


> [!IMPORTANT]
> 键盘按键对应的是动作策略在 `mimic_policies` 中的索引顺序。
>
> 修改动作策略的排列顺序后，需要同步检查键盘映射和状态机中的动作索引。

---

## 策略切换状态机

本项目使用的自定义状态机位于：

```text
robojudo/pipeline/mjlab_loco_action_pipeline.py
```

对应的状态机类为：

```python
MjlabLocoActionPipeline
```

当前状态包括：

| 状态 | 含义 |
| --- | --- |
| `LOCO` | 运行常规行走策略 |
| `ACTION` | 运行选定的动作策略 |
| `RETURN` | 平滑过渡回行走策略 |
| `STABILIZE` | 动作结束后使用 AMP 策略进行稳定控制 |
| `RECOVERY` | 手动进入 AMP 恢复模式 |

### 状态切换规则

- 按下 `1`、`2`、`4`、`5`、`6` 或 `7`：
  - 执行对应动作；
  - 动作结束后进入 `RETURN`；
  - 平滑切换回行走策略。

- 按下 `3`：
  - 执行 `rear_straight_punch`；
  - 动作结束后进入 `STABILIZE`；
  - 使用 AMP 策略进行稳定控制；
  - 随后进入 `RETURN` 并切换回行走策略。

- 按下 `9`：
  - 手动进入 `RECOVERY`；
  - 使用 AMP 策略恢复机器人姿态。

- 按下 `0`：
  - 当机器人已经保持直立且姿态稳定时，退出 `RECOVERY` 或 `STABILIZE`；
  - 进入 `RETURN` 并平滑切换回行走策略。

为避免动作相互打断，新的动作触发仅在 `LOCO` 状态下有效。

在以下状态中，不接受新的动作切换请求：

- `ACTION`
- `RETURN`
- `STABILIZE`
- `RECOVERY`

当前推荐配置为：

```text
g1_mjlab_loco_right_overhand_post_action_only
```

## 键盘控制

启动 MuJoCo 仿真窗口后，需要将输入焦点切换到负责捕获键盘输入的终端窗口。

### 行走控制

| 按键 | 控制指令 |
| --- | --- |
| `W` | 向前移动，速度为 `+0.5 m/s` |
| `S` | 向后移动，速度为 `-0.3 m/s` |
| `A` | 向左横移，速度为 `+0.2 m/s` |
| `D` | 向右横移，速度为 `-0.2 m/s` |
| `Q` | 向左转动，角速度为 `+0.3 rad/s` |
| `E` | 向右转动，角速度为 `-0.3 rad/s` |
| `Space` | 将速度指令清零 |

速度指令采用锁存方式：

- 按下方向键后，速度指令会持续生效；
- 松开按键不会自动停止机器人；
- 需要按下 `Space` 将速度指令清零。

### 动作与状态控制

动作和状态命令在按键释放时触发。

| 按键 | 功能 |
| --- | --- |
| `1` | 执行 `right_overhand` |
| `2` | 执行 `back_kick` |
| `3` | 执行 `rear_straight_punch`，随后进入稳定状态 |
| `4` | 执行 `left_jab` |
| `5` | 执行 `right_cross` |
| `6` | 执行 `left_front_kick` |
| `7` | 执行 `spin_kick` |
| `9` | 手动进入 AMP 恢复模式 |
| `0` | 在机器人稳定直立后返回行走模式 |
| `` ` `` | 重置仿真环境或执行 reborn |
| `Esc` | 关闭程序 |

---

## 快速开始

### 运行 MuJoCo sim2sim

进入项目目录：

```bash
cd /root/gpufree-data/RoboJuDo
```

激活 Conda 环境：

```bash
conda activate robojudo
```

运行自定义控制流程：

```bash
python scripts/run_pipeline.py \
  -c g1_mjlab_loco_right_overhand_post_action_only
```

### 推荐测试流程

1. 启动仿真程序。
2. 按下 `Space`，确保初始速度指令为零。
3. 分别按下 `W`、`A`、`S`、`D`、`Q` 和 `E`，验证行走控制。
4. 再次按下 `Space`，停止机器人。
5. 依次按下 `1` 至 `7`，分别测试动作策略。
6. 测试动作过程中不要连续触发多个动作按键。
7. 如果机器人跌倒，按下 `9` 进入 AMP 恢复模式。
8. 等待机器人恢复直立并保持稳定。
9. 按下 `0` 返回行走模式。

---

## 开发环境配置

建议使用 Python 3.11 创建独立 Conda 环境：

```bash
conda create -n robojudo python=3.11 -y
conda activate robojudo
```

以可编辑模式安装项目：

```bash
pip install -e .
```

### CPU 版本 PyTorch

在不需要 CUDA 的环境中，可以安装 CPU 版本 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### 已知可选依赖问题

完整测试套件中的部分模块可能因为缺少可选依赖而导入失败：

- 部分控制器模块依赖 `phc` 和 `redis`；
- 真实机器人环境依赖 `unitree_sdk2py` 或 `unitree_cpp`。

这些依赖不是当前 MuJoCo sim2sim 流程的必需项，但在部署到真实机器人之前必须正确安装和配置。

---

## 真实机器人部署说明

当前配置使用：

```python
G1MujocoEnvCfg
```

因此它属于仿真环境配置，不能直接在真实 Unitree G1 上运行。

真实机器人部署前，至少需要完成以下工作。

### 1. 安装 Unitree SDK

确保以下 SDK 中至少一个已经正确安装并能够导入：

```text
unitree_sdk2py
unitree_cpp
```

同时确认 SDK 版本与机器人系统版本兼容。

### 2. 配置 DDS 与网络

控制程序需要运行在以下设备之一：

- Unitree G1 机器人计算机；
- 与机器人处于同一 DDS 网络的现场控制计算机；
- 与机器人通过有线网络可靠连接的外部计算机。

需要确认：

- 控制计算机与机器人处于正确网段；
- DDS 网络接口配置正确；
- 控制数据包可以稳定发送和接收；
- 网络延迟和丢包率满足实时控制要求。

### 3. 检查关节映射

真实机器人必须与策略使用相同的受控自由度布局，包括：

- 腿部关节；
- 腰部关节；
- 手臂关节；
- 腕部关节。

必须逐项核对：

- 关节数量；
- 关节名称；
- 关节顺序；
- 关节正负方向；
- 关节零位定义；
- 关节限位。

### 4. 标定控制参数

必须检查并重新标定：

- PD 增益；
- 力矩上限；
- 位置指令上限；
- 速度指令上限；
- 动作缩放系数；
- 默认关节姿态；
- 控制周期；
- 策略推理频率。

不要直接将仿真环境中的高增益参数原样应用到真实机器人。

### 5. 检查状态估计

需要确认策略实际使用的状态信息来源，包括：

- IMU 姿态；
- IMU 角速度；
- 关节位置；
- 关节速度；
- 基座速度；
- 里程计数据；
- 接触状态。

AMP 恢复与稳定模块使用的姿态阈值，需要根据真实机器人的 IMU 和里程计数据重新标定。

### 6. 配置安全机制

真实机器人测试必须独立准备：

- 硬件急停；
- 软件急停；
- 阻尼模式；
- 失联保护；
- 指令超时保护；
- 关节限位保护；
- 力矩限制；
- 跌倒检测；
- 控制程序异常退出后的安全状态。

AMP 恢复策略只能作为策略层辅助功能，不能替代硬件急停、阻尼模式或人工保护。


## 当前项目状态

### 已完成

- MuJoCo 环境中的 G1 行走策略加载；
- 7 个动作策略集成；
- 键盘行走控制；
- 动作运行时切换；
- 动作结束后的平滑返回；
- AMP 手动恢复；
- 指定动作结束后的 AMP 稳定流程；
- 相关回归测试。

### 尚未完成或仍需验证

- 真实 Unitree G1 环境适配；
- Unitree SDK 和 DDS 通信验证；
- 真实机器人关节映射；
- 真实机器人控制增益标定；
- 硬件急停与失联保护；
- AMP 恢复策略的实机有效性验证；
- 真实机器人上的动作安全性验证。

---

## 许可证与上游项目

本项目基于 HansZ8 开源的 RoboJuDo 项目进行修改：

- 上游项目：[HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo)

发布本项目的修改版本时，应保留上游项目的许可证文件、版权声明和项目来源说明。

本仓库中的自定义代码、模型权重和动作数据是否允许公开发布，应根据各自的数据来源、训练资源和许可证要求单独确认。

---

## 免责声明

本项目涉及人形机器人的高动态运动控制。

动作策略、恢复策略和仿真测试结果不能保证在真实机器人上具有相同表现。错误的关节映射、控制增益、动作缩放、网络配置或状态估计可能导致机器人跌倒、设备损坏或人员受伤。

真实机器人测试必须在具备急停、安全绳、支撑装置和现场保护人员的条件下进行。
````
