from game_glory import *
from render import *
import time
import torch
render = Render()
env = PolytopiaEnv()
obs = env.reset()

render.render(env.get_obs())
steps = [95,1,71,121,26,1,38,38,2,15,2,121,106,2,84,2,71,1,59,121,38,1,49,49,1,59,121,59,1,49,95,5,121,49,1,59,121,121,59,3]
steps = [84, 2, 96, 2, 95, 1 ,96, 37]
for i in range(len(steps)):
    mask = env.get_mask()
    options = []
    #print(env.get_obs(1)[71][11], env.get_obs(1)[71][6], env.get_obs(1)[71][0], env.turn, env.step_phase)
    for j in range(len(mask)):
        if mask[j] == 1:
            options.append(j)
    print(options)
    env.step(steps[i])
    print(steps[i])
    obs = env.get_obs(1)
    render.render(obs)
    time.sleep(0.1)
print(env.get_turn_and_points())
time.sleep(180)
