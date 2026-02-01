import random

from game_glory import PolytopiaEnv


def pick_action(env, pass_chance=0.1):
    mask = env.get_mask()
    valid_actions = [idx for idx, allowed in enumerate(mask) if allowed]

    if env.step_phase == 0:
        if not valid_actions:
            return 121
        if random.random() < pass_chance:
            return 121
        return random.choice(valid_actions)

    if not valid_actions:
        return 0

    return random.choice(valid_actions)


def run_random_game(max_steps=500, pass_chance=0.1):
    env = PolytopiaEnv()
    env.reset()

    done = False
    step_count = 0
    while not done and step_count < max_steps:
        action = pick_action(env, pass_chance=pass_chance)
        _, _, done, info = env.step(action)
        step_count += 1

    return {
        "steps": step_count,
        "done": done,
        "info": info if done else {},
    }


if __name__ == "__main__":
    result = run_random_game()
    print(result)
