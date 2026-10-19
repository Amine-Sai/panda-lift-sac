import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from sac.train_sac import (
    make_env,
    HORIZON,
    CONTROL_FREQ,
    REPLAY_SIZE,
)
from sac.sac_torch import Agent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["native", "sparse", "pbrs"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=HORIZON)

    args = parser.parse_args()

    run_dir = Path("runs_sac") / f"{args.mode}_s{args.seed}"

    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "best.pt"
    )

    output = (
        Path(args.output)
        if args.output
        else Path("results/videos")
        / f"{args.mode}_s{args.seed}.mp4"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(
        args.mode,
        has_offscreen_renderer=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    try:
        agent = Agent(
            input_dims=env.observation_space.shape,
            env=env,
            n_actions=env.action_space.shape[0],
            max_size=REPLAY_SIZE,
            seed=args.seed,
        )

        agent.load(str(checkpoint))

        # Match the evaluation environment's seeded reset approach.
        np.random.seed(args.eval_seed)
        obs = env.reset()

        success_step = None
        steps_taken = 0

        def render_frame():
            return env.unwrapped.sim.render(
                camera_name=args.camera,
                width=args.width,
                height=args.height,
            )[::-1].copy()

        with imageio.get_writer(
            str(output),
            fps=CONTROL_FREQ,
            codec="libx264",
        ) as video:

            video.append_data(render_frame())

            for step in range(args.steps):
                action = agent.choose_action_deterministic(obs)

                obs, _, done, _ = env.step(action)

                video.append_data(render_frame())
                steps_taken = step + 1

                if (
                    success_step is None
                    and env.unwrapped._check_success()
                ):
                    success_step = steps_taken

                if done:
                    break

        print(f"Checkpoint: {checkpoint}")
        print(f"Video saved: {output}")
        print(f"Steps recorded: {steps_taken}")
        print(f"Lift success: {success_step is not None}")

        if success_step is not None:
            print(f"First successful lift at step: {success_step}")

    finally:
        env.close()


if __name__ == "__main__":
    main()