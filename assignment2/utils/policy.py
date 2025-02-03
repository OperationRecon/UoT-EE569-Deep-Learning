import numpy as np
class Epsilon_Greedy_Policy():
    '''Defines the Epsilon Greedy Policy'''
    def __init__(self, epsilon=0.98, decay=0.998):
        self.epsilon = epsilon
        self.current_epsilon = epsilon
        self.decay = decay
    
    def select_action(self):
        '''return wether to exploit(true) or explore(false)'''
        if np.random.rand() < self.current_epsilon:
            # explore
            exploit =  False
        else:
            # exploit
            exploit = True
        
        # update epsilon
        self.current_epsilon *= self.decay
        return exploit
        