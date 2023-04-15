from gym import Env
from gym.spaces import Discrete, Box
from numpy import random

class GridWorldEnv(Env):
    '''
    Class defining a general Grid World environment.

    The grid is 'height' by 'width', and each square corresponds to an integer index between 0 and height*width-1.
    The agent starts at 0 and tries to go to height*width-1.
    If the step takes the agent out of bounds, the agent stays in its place.
    At each non-terminating step taken, the agent receives a reward of -12 or +10 with equal probability.
    Upon reaching the target state, the next step the agent takes will give it a reward of +5, and the episode will end.

    Observation space: {0, 1, ..., height*width-1}, corresponding to the squares in the grid
    Action space: {0, 1, 2, 3}, corresponding to {UP, LEFT, RIGHT, DOWN}
    '''

    UP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3

    def __init__(self, height, width):
        self.action_space = Discrete(4)
        self.height = height
        self.width = width
        self.observation_space = Discrete(self.height * self.width)
        # initial state
        self.state = 0
        self.end_state = self.height * self.width - 1

    def step(self, action):
        '''
        Function applying the action to current state.
        If the action makes the agent go out of bounds, the agent remains in the same place.
        '''

        terminated = False

        # sample reward randomly
        if random.binomial(n=1, p=0.5, size=1) == 0:
            reward = -12
        else:
            reward = 10

        # check if currently at target
        if self.state == self.end_state:
            reward = 5
            terminated = True
        
        # otherwise, check action and apply if valid
        elif action == self.UP:
            if self.state + self.width <= self.end_state:
                self.state += self.width
        
        elif action == self.LEFT:
            if self.state % self.width != 0:
                self.state -= 1

        elif action == self.RIGHT:
            if self.state % self.width != self.width - 1:
                self.state += 1

        elif action == self.DOWN:
            if self.state >= self.width:
                self.state -= self.width

        information = {}
        
        return self.state, reward, terminated, information

    def reset(self):
        '''
        Function that resets the environment after termination.
        '''
        self.state = 0
        information = {}

        return self.state, information

    def render(self):
        pass