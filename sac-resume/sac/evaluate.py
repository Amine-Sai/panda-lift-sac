import numpy as np


def evaluate(agent, env, seeds):
    state = np.random.get_state()
    successes = 0
    try:
        for s in seeds:
            np.random.seed(s)
            obs = env.reset()
            done = False
            while not done:
                obs, _, done, _ = env.step(agent.choose_action_deterministic(obs))
                if env.unwrapped._check_success():
                    successes += 1
                    break
    finally:
        np.random.set_state(state)
    return successes / len(seeds)
