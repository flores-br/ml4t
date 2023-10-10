""""""  		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Bryan Flores 		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: bflores9	  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903848430  		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	

import numpy as np


class QLearner(object):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This is a Q learner object.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param num_states: The number of states to consider.  		  	   		  		 			  		 			 	 	 		 		 	
    :type num_states: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param num_actions: The number of actions available..  		  	   		  		 			  		 			 	 	 		 		 	
    :type num_actions: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type alpha: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type gamma: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type rar: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  		 			  		 			 	 	 		 		 	
    :type radr: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  		 			  		 			 	 	 		 		 	
    :type dyna: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.98, radr=0.999, dyna=0, verbose=False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.Q = np.zeros((num_states, num_actions)) # Q table
        self.s = 0 # current state
        self.a = 0 # current action
        self.t = np.zeros((num_states, num_actions, num_states)) # transition probability table
        self.r = np.zeros((num_states, num_actions)) # reward table
        self.history = set() # state-action history

    def querysetstate(self, s):
        self.s = s
        if np.random.random() < self.rar:
            self.a = np.random.randint(0, self.num_actions)
        else:
            self.a = np.argmax(self.Q[s])

        if self.verbose:
            print(f"State: {s}, \nAction: {self.a}")

        return self.a

    def query(self, s_prime, r):
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
                  r + self.gamma * np.max(self.Q[s_prime]))

        if self.dyna > 0:
            self.history.add((self.s, self.a))
            self.t[self.s, self.a, s_prime] += 1
            self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + self.alpha * r
            self.update_dyna()

        self.a = self.querysetstate(s_prime)
        self.rar *= self.radr
        self.s = s_prime

        if self.verbose:
            print(f"State: {s_prime} \nAction: {self.a} \nReward: {r}")

        return self.a

    def update_dyna(self):
        for _ in range(self.dyna):
            state, action = list(self.history)[np.random.choice(len(self.history))]
            s_prime = np.argmax(self.t[state, action])
            r = self.r[state, action]
            self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (r + self.gamma * np.max(self.Q[s_prime]))

    def author(self):
        return "bflores9"
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		  		 			  		 			 	 	 		 		 	
    print("Remember Q from Star Trek? Well, this isn't him")