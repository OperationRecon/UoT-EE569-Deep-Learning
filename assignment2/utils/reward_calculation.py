import numpy as np

def calculate_reward(env, r):
    '''Checks the enviroment for certain conditions in order to modify the reward'''
    speed = [i.hull.linearVelocity.length for i in env.cars]

    # reward speed and progression
    r +=  (-14 + 1 * np.array(speed)) * 0.2 + (np.array(env.tile_visited_count) * np.array(speed) > 6 * 0.1)

    # heavily penalise cars for going off track or driving too slowly
    r = [r[i] -18 if  env.driving_on_grass[i] else r[i] for i in range(len(r))]

    return  r
