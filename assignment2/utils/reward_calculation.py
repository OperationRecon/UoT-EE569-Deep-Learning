import numpy as np

def calculate_reward(env, r):
    '''Checks the enviroment for certain conditions in order to modify the reward'''

    speed = np.array([i.hull.linearVelocity.length for i in env.cars])

    on_grass = np.array(env.driving_on_grass, np.int8)

    backwards = np.array(env.backwards_flag)

    # reward speed and forward progression
    r = r + (((-4 + speed) * 0.002 + np.array(env.tile_visited_count) * np.where(speed > 4, 1,0) * 0.008)) * (1 - on_grass) * (1 - backwards)

    # penalise cars for going off track
    r += -0.4 * on_grass


    return  r
