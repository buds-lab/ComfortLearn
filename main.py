from pathlib import Path
import sys
import yaml

# import torch
from comfortlearn import ComfortLearn
from agent import Baseline


# load config file from CLI
with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# check for seed and make sure it's an int
ext_seed = None
if len(sys.argv) > 2:
    ext_seed = int(sys.argv[2])

# load environment from files
env_params = {
    "experiment_name": config["config_name"],
    "seed": config["seed"] if ext_seed is None else ext_seed,
    "data_path": Path(config["data_path"]),
    "num_new_occupants": config["num_new_occupants"],
    "zone_selection": config["zone_selection"],
    "occupant_timing": config["occupant_timing"],
    "occupant_tolerance": config["occupant_tolerance"],
    "occupant_tol_file": config["occupant_tol_file"],
    "occupant_preference": config["occupant_preference"],
    "occupant_background": config["occupant_background"],
    "occupant_pcm": config["occupant_pcm"],
    "zone_attributes": config["zone_attributes"],
    "weather_file": config["weather_file"],
    "zones_states_actions": config["zones_states_actions"],
    "simulation_period": (0, config["max_steps"] - 1),
    "cost_function": config["cost_function"],
    "central_agent": config["central_agent"],
    "verbose": config["verbose"],
}

# instantiate environment and get observations, actions, and zone
env = ComfortLearn(**env_params)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# instantiate control agent modify here for different agents for the controller
params_agent = {
    "observation_spaces": observations_spaces,
    "action_spaces": actions_spaces,
}

if config["agent_type"] == "Baseline":
    agent = Baseline(**params_agent)
else:
    print(f"Sorry! Agent {config['agent_type']} hasn't been implemented yet")

# start simulation
state = env.reset()
done = False

actions = agent.select_action(state)

while not done:
    next_state, reward, done = env.step(actions)
    action_next = agent.select_action(next_state)
    action = action_next
