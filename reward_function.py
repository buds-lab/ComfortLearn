import numpy as np

# TODO: reward function for multi agent


def reward_function_sa(alpha, beta, thermal_comfort, electricity_demand):
    reward_ = alpha * thermal_comfort + beta * electricity_demand

    return reward_
