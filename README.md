# Soft Actor-Critic for Robotic Manipulation

A reinforcement-learning project that trains a **Soft Actor-Critic (SAC)** agent to lift a cube with a Panda robot in the [`robosuite`](https://robosuite.ai/) `Lift` task, simulated with MuJoCo. The project compares reward formulations, with the built-in robosuite shaped reward as the main demonstrated baseline.

## Demonstration

https://github.com/user-attachments/assets/f260a0cb-2eec-4761-b219-0030213f3a54


**[Open the native-reward policy recording](results/videos/demonstration.mp4)**

## Current result

In the reported experiment of approximately 1,000 training episodes per reward mode, the `native` run reached **about 98% success on a training-time evaluation**. The sparse-reward and potential-based reward-shaping (PBRS) runs were below 10% in the runs observed so far.

These figures are a snapshot of the current experiments, not a statistically conclusive comparison: the results are seed-dependent, and the strongest native number is a best periodic evaluation rather than a guarantee of held-out performance. Check each run's `final_eval.json` for its separate held-out result before quoting final numbers.

The root-level `curves.png` provides a quick view of evaluation success over environment steps:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6790824a-bcb8-4f04-b16a-5caf783458be" alt="Evaluation success curves" width="650">
</p>

## Task and observations

- **Environment:** robosuite `Lift`, Panda robot, MuJoCo.
- **Controller:** `OSC_POSITION`.
- **Observation:** low-dimensional state, using `robot0_eef_pos`, `robot0_gripper_qpos`, and `object-state`. Camera observations are disabled during training.
- **Action:** continuous robot-control actions from the robosuite Gym wrapper.
- **Evaluation:** deterministic policy actions in a sparse-reward Lift environment, using fixed evaluation seeds.

The environment's success predicate is checked during each episode. A success means the predicate was reached at least once; it does not independently verify that the cube remained stably lifted for a sustained period.

## Reward modes

Select a mode with `--mode`:

| Mode | Reward configuration | Purpose |
|---|---|---|
| `native` | robosuite Lift with `reward_shaping=True` | Main baseline and demonstrated result. |
| `sparse` | robosuite Lift with `reward_shaping=False` | Tests learning from the task's sparse reward. |
| `pbrs` | `CustomLift` with sparse base reward and potential-based shaping at scale `1.0` by default | Explores an auxiliary distance/grasp/lift potential. |

In `CustomLift`, the potential increases as the end effector approaches the cube and adds terms when the cube is grasped and raised. The shaping contribution uses the discounted potential difference. These variants are exploratory comparisons; the current results do not show them matching the native baseline under the tested budget.

## Installation

```bash
git clone https://github.com/Amine-Sai/sac-learning.git
cd sac-learning

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the core imports:

```bash
python -c "import torch, numpy, mujoco, robosuite; print('torch', torch.__version__); print('numpy', numpy.__version__); print('mujoco', mujoco.__version__); print('robosuite', robosuite.__version__)"
```

For CUDA acceleration, install a PyTorch build compatible with your operating system and CUDA setup. PyTorch's installation command may differ by platform; use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) if the pinned build is unsuitable for your machine. Training also works on CPU, but can be slower.

## Train

Run one reward mode at a time from the repository root:

```bash
python -m sac.train_sac --mode native --seed 0 --episodes 1000
python -m sac.train_sac --mode sparse --seed 0 --episodes 1000
python -m sac.train_sac --mode pbrs --seed 0 --episodes 1000

```

`--episodes` is the **total target episode count**, not the number of extra episodes to add after resuming. The default in the training script is 2,000 episodes. Change `--seed` to launch a separate run; the output directory includes both the reward mode and seed.

The principal training settings are defined in `sac/train_sac.py`: a 500-step episode horizon, control frequency of 20 Hz, 5,000 warm-up steps, replay-buffer capacity of 1,000,000 transitions, and periodic evaluation every 25,000 environment steps.

## Checkpoints and resuming

Each run writes to `runs_sac/<mode>_s<seed>/`. The main files are:

| File | Purpose |
|---|---|
| `best.pt` | Policy and training state from the best periodic evaluation so far. |
| `latest.pt` | Most recent periodic checkpoint, including resume metadata. |
| `buffer.npz` | Saved replay buffer, written every four periodic evaluations. |
| `eval.csv` | Periodic evaluation success as two headerless columns: environment steps and success fraction. |
| `final_eval.json` | Best periodic evaluation plus the final held-out evaluation metrics, if the run completed that stage. |
| TensorBoard event file(s) | Training return, episode success, grasp indicator, and evaluation metrics. |

Resume the same mode and seed by passing `--resume`:

```bash
python -m sac.train_sac --mode native --seed 0 --episodes 1000 --resume
```

This example targets 1,000 total episodes. The checkpoint must exist under `runs_sac/native_s0/`. Preserve both `latest.pt` and `buffer.npz` for the closest available continuation; the replay buffer is saved less frequently than the model checkpoint, so it may represent an earlier point in the run. Checkpoints are periodic, so an interruption can lose work since the most recent save. Keep large checkpoint and replay-buffer files out of ordinary Git commits; archive them separately if needed.

## Evaluation protocol

During training, the best checkpoint is selected by success rate over a fixed set of 50 evaluation seeds (`1000`–`1049`). At the end of training, the script loads `best.pt` and evaluates it over a separate set of 100 seeds (`2000`–`2099`), writing the held-out score to `final_eval.json`.

For a fairer reward-mode comparison, run each mode with multiple independent training seeds and compare held-out success, not training return alone. Training returns are not directly comparable when each mode uses a different reward scale or reward function.

## Plotting

From the repository root, regenerate the quick evaluation curve with:

```bash
python plot_curves.py
```

You can inspect training metrics interactively with TensorBoard:

```bash
tensorboard --logdir runs_sac
```

## Repository structure

```text
sac-learning/
├── sac/
│   ├── train_sac.py       # training loop, reward-mode selection, checkpointing
│   ├── sac_torch.py       # SAC agent and update rules
│   ├── networks.py        # actor and critic networks
│   ├── buffer.py          # replay buffer
│   ├── evaluate.py        # deterministic evaluation
│   └── custom_lift.py     # potential-based shaping variant
├── runs_sac/              # per-mode/per-seed training output
├── plot_curves.py         # quick evaluation-curve plot
├── curves.png             # saved learning-curve figure
├── results/videos/        # demonstration videos (if added to the repository)
└── requirements.txt
```

## References

- Haarnoja et al., [*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*](https://arxiv.org/abs/1801.01290).
- robosuite documentation: [https://robosuite.ai/](https://robosuite.ai/).
- Ng, Harada, and Russell, [*Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
