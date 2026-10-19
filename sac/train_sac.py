import argparse
import json
import os
from collections import deque

import numpy as np
import torch as T
import robosuite as suite
from sac.custom_lift import CustomLift
from robosuite.environments.manipulation.lift import Lift
from robosuite.wrappers import GymWrapper
from torch.utils.tensorboard import SummaryWriter

from sac.sac_torch import Agent
from sac.evaluate import evaluate


ENV_NAME = "Lift"
ROBOT = "Panda"
HORIZON = 500
CONTROL_FREQ = 20

N_EPISODES = 2000
WARMUP_STEPS = 5000
REPLAY_SIZE = 1_000_000
EVAL_EVERY_STEPS = 25_000
EVAL_SEEDS = list(range(1000, 1050))
FINAL_EVAL_SEEDS = list(range(2000, 2100))
BUFFER_SAVE_EVERY_EVALS = 4

OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "object-state",
]


def make_env(reward_mode="sparse", horizon=HORIZON, control_freq=CONTROL_FREQ, **env_kwargs):
    assert reward_mode in ("sparse", "native", "pbrs", "pbrs10")

    kwargs = dict(
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="OSC_POSITION"),
        use_camera_obs=False,
        use_object_obs=True,
        has_offscreen_renderer=False,
        horizon=horizon,
        control_freq=control_freq,
        **env_kwargs,
    )
    if reward_mode in ("pbrs", "pbrs10"):
        env = CustomLift(reward_shaping=False, pbrs_scale=10.0 if reward_mode == "pbrs10" else 1.0, **kwargs)
    else:
        env = Lift(reward_shaping=(reward_mode == "native"), **kwargs)
    return GymWrapper(env, keys=OBS_KEYS)


def train(
    reward_mode,
    seed,
    n_episodes=N_EPISODES,
    warmup_steps=WARMUP_STEPS,
    resume=False,
):
    np.random.seed(seed)
    T.manual_seed(seed)

    env = make_env(reward_mode)
    eval_env = make_env("sparse")
    env.action_space.seed(seed)

    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        max_size=REPLAY_SIZE,
        seed=seed,
    )


    run_dir = f"runs_sac/{reward_mode}_s{seed}"
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    total_steps = 0
    success_count = 0
    next_log_step = 5000
    next_eval_step = EVAL_EVERY_STEPS
    best_eval = -1.0
    n_evals = 0
    start_episode = 0

    if resume:
        extra = agent.load(os.path.join(run_dir, "latest.pt"))
        buf_path = os.path.join(run_dir, "buffer.npz")
        if os.path.exists(buf_path):
            agent.memory.load(buf_path)
        total_steps = extra["total_steps"]
        start_episode = extra["episode"]
        best_eval = extra["best_eval"]
        next_eval_step = extra["next_eval_step"]
        n_evals = extra["n_evals"]
        success_count = extra["success_count"]
        np.random.set_state(extra["np_state"])
        agent.memory.rng.bit_generator.state = extra["buf_rng"]

    score_history = deque(maxlen=100)
    success_history = deque(maxlen=100)

    for episode in range(start_episode, n_episodes):
        obs = env.reset()
        done = False
        success_any = False
        success_steps = 0
        score = 0.0
        grasp = False

        while not done:
            total_steps += 1

            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(obs)

            next_obs, reward, done, _ = env.step(action)

            ok = bool(env.unwrapped._check_success())
            success_any |= ok
            success_steps += int(ok)
            grasp |= bool(env.unwrapped._check_grasp(
                gripper=env.unwrapped.robots[0].gripper,
                 object_geoms=env.unwrapped.cube,
                 )
)

            agent.remember(
                obs,
                action,
                reward,
                next_obs,
                False,
            )

            obs = next_obs
            score += reward

            if total_steps >= warmup_steps:
                agent.learn()

        success_count += int(success_any)
        score_history.append(score)
        success_history.append(int(success_any))
        average_score = np.mean(score_history)

        writer.add_scalar("train/return", score, total_steps)
        writer.add_scalar("train/success", int(success_any), total_steps)
        writer.add_scalar("train/success_steps", success_steps, total_steps)
        writer.add_scalar("train/grasp", int(grasp), total_steps)

        if total_steps >= next_eval_step:
            eval_success = evaluate(agent, eval_env, EVAL_SEEDS)
            writer.add_scalar("eval/success", eval_success, total_steps)
            with open(os.path.join(run_dir, "eval.csv"), "a") as f:
                f.write(f"{total_steps},{eval_success}\n")
            if eval_success > best_eval:
                best_eval = eval_success
                agent.save(os.path.join(run_dir, "best.pt"))
            next_eval_step += EVAL_EVERY_STEPS
            n_evals += 1
            agent.save(os.path.join(run_dir, "latest.pt"), extra={
                "total_steps": total_steps, "episode": episode + 1, "best_eval": best_eval,
                "next_eval_step": next_eval_step, "n_evals": n_evals, "success_count": success_count,
                "np_state": np.random.get_state(), "buf_rng": agent.memory.rng.bit_generator.state})
            if n_evals % BUFFER_SAVE_EVERY_EVALS == 0:
                agent.memory.save(os.path.join(run_dir, "buffer.npz"))

        if episode % 10 == 0:
             success_rate = np.mean(success_history)
             print(
                f"Ep {episode:4d} | "
                f"Steps {total_steps:7d} | "
                f"Avg {average_score:7.2f} | "
                f"Success {success_rate:.1%}"
            )

        if total_steps >= next_log_step:
            if total_steps >= warmup_steps:
                print(
                    f"Step {total_steps} | "
                    f"Q1 {agent.q1.mean().item():.2f} | "
                    f"Q2 {agent.q2.mean().item():.2f} | "
                    f"LogP {agent.log_probs.mean().item():.2f} | "
                    f"Alpha {agent.alpha:.4f}"
                )

            next_log_step += 5000

    best_path = os.path.join(run_dir, "best.pt")
    if os.path.exists(best_path):
        agent.load(best_path)
        heldout = evaluate(agent, eval_env, FINAL_EVAL_SEEDS)
        with open(os.path.join(run_dir, "final_eval.json"), "w") as f:
            json.dump({"mode": reward_mode, "seed": seed, "best_eval_success": best_eval,
                       "heldout_success": heldout, "heldout_episodes": len(FINAL_EVAL_SEEDS)}, f)
        print(f"Held-out success (best.pt): {heldout:.1%}")

    eval_env.close()
    env.close()
    writer.close()

    print(
        f"\nFinished {reward_mode} training | "
        f"Steps: {total_steps} | "
        f"Successes: {success_count}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sparse", "native", "pbrs", "pbrs10"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    train(a.mode, a.seed, a.episodes, resume=a.resume)


if __name__ == "__main__":
    main()