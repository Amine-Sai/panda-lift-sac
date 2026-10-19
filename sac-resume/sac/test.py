import sys
import time

import robosuite as suite
from sac.train_sac import make_env, HORIZON
from robosuite.wrappers import GymWrapper

from sac.sac_torch import Agent


N_EVAL_EPISODES = 10


env = make_env(
    reward_mode="sparse",
    has_renderer=True,
    render_camera="frontview",
    control_freq=30,
)

print("Observation shape:", env.observation_space.shape)
print("Action shape:", env.action_space.shape)

agent = Agent(
    input_dims=env.observation_space.shape,
    env=env,
    n_actions=env.action_space.shape[0],
)

try:
    agent.load(sys.argv[1])
    print("Successfully loaded trained model.")
except FileNotFoundError:
    print("Could not find model checkpoints.")
    env.close()
    raise

success_count = 0

for ep in range(N_EVAL_EPISODES):
    obs = env.reset()
    done = False
    success = False
    score = 0.0

    print(f"\n--- Starting Episode {ep + 1} ---")

    while not done:
        action = agent.choose_action_deterministic(obs)

        next_obs, reward, done, info = env.step(action)

        env.render()
        time.sleep(0.01)

        obs = next_obs
        score += reward

        success = success or bool(env.unwrapped._check_success())

        if success:
            break

    success_count += int(success)
    print(
        f"Episode {ep + 1} finished | "
        f"Score: {score:.3f} | "
        f"Success: {success}"
    )

success_rate = 100 * success_count / N_EVAL_EPISODES

print("\n" + "=" * 35)
print(f"Episodes:       {N_EVAL_EPISODES}")
print(f"Successful:     {success_count}")
print(f"Success rate:   {success_rate:.1f}%")
print("=" * 35)

env.close()