from gym import Env
from gym.spaces import Discrete, Box
from numpy import random

class GridWorld3x3Env(Env):
    '''
    Class defining a Grid World environment similar to the one used in Hado van Hasselt's Double Q-learning paper.

    The grid is 3x3, and each square corresponds to an integer index between 0 and 8.
    The agent starts at 0 and tries to go to 8.
    If the step takes the agent out of bounds, the agent stays in its place.
    At each non-terminating step taken, the agent receives a reward of -12 or +10 with equal probability.
    Upon reaching the target state, the next step the agent takes will give it a reward of +5, and the episode will end.

    Observation space: {0, 1, ..., 8}, corresponding to the squares in the grid
    Action space: {0, 1, 2, 3}, corresponding to {UP, LEFT, RIGHT, DOWN}
    '''

    UP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    GRID_WIDTH = 3

    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.GRID_WIDTH * self.GRID_WIDTH)
        # initial state
        self.state = 0
        self.end_state = self.GRID_WIDTH * self.GRID_WIDTH - 1

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
        if self.state == 8:
            reward = 5
            terminated = True
        
        # otherwise, check action and apply if valid
        elif action == self.UP:
            if self.state + self.GRID_WIDTH <= self.end_state:
                self.state += self.GRID_WIDTH
        
        elif action == self.LEFT:
            if self.state % self.GRID_WIDTH != 0:
                self.state -= 1

        elif action == self.RIGHT:
            if self.state % self.GRID_WIDTH != 2:
                self.state += 1

        elif action == self.DOWN:
            if self.state >= self.GRID_WIDTH:
                self.state -= self.GRID_WIDTH

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