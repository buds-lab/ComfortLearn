# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from sklearn.linear_model import LinearRegression
# from common.rl import *
# from common.preprocessing import *


class Baseline:
    """No real controller. Only historical operational data is executed"""

    def __init__(self, observation_spaces, action_spaces):
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.time_step = 0

    def select_action(self, action):
        return None
