import numpy as np

def calculate_reward(env, r):
    '''Checks the enviroment for certain conditions in order to modify the reward'''
    speed = np.array([i.hull.linearVelocity.length for i in env.cars])

    on_grass = np.array(env.driving_on_grass, np.int8)

    # reward speed and progression
    r = r + (((-4 + speed) * 0.0002 + np.array(env.tile_visited_count) * np.where(speed > 4, 1,0) * 0.00008)) * (1 - on_grass)

    # heavily penalise cars for going off track or driving too slowly
    r += -0.2 * on_grass


    return  r
