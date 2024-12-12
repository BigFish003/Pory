from game_glory import *
from render import *
import time
import torch
render = Render()
env = PolytopiaEnv()
obs = env.reset()

render.render(env.get_obs())
env.step([95,1,83])
time.sleep(10)