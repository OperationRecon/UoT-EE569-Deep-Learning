import numpy as np
class Epsilon_Greedy_Policy():
    '''Defines the Epsilon Greedy Policy'''
    def __init__(self, epsilon=1, min_eps = 0.1, decay=0.992):
        self.epsilon = epsilon
        self.current_epsilon = epsilon
        self.decay = decay
        self.min_eps = min_eps
    
    def select_action(self):
        '''return wether to exploit(true) or explore(false)'''
        if np.random.rand() < max(self.min_eps, self.current_epsilon):
            # explore
            exploit =  False
        else:
            # exploit
            exploit = True
        
        # update epsilon
        
        return exploit
    
    def update_epsilon(self,):
        '''update epsilon'''
        self.current_epsilon *= self.decay
        