import warnings
import os
import logging
import json
import random
from collections import defaultdict
import gym
import numpy as np
import pandas as pd
from numpy.random import choice
from gym.utils import seeding
from common.utils import load_variable, save_variable, tp_dist
from energy_models import Zone


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter("ignore", ResourceWarning)


class Occupant:
    """
    Occupant object based on available data. The dataframes with metadata,
    phisiological, and environmental data will be loaded and then filtered
    based on the `user_ids`
    """

    def __init__(self, user_id, pcm):
        self.user_id = user_id
        self.pcm = pcm
        self.tp_real = defaultdict(list)  # {zone_uid: [thermal preference ground truth]}
        self.tp_pred = defaultdict(list)  # {zone_uid: [thermal preference prediction]}
        self.t_in = defaultdict(list)  # {zone_uid: [t_in]}


def occupant_loader(
    num_new_occupants,
    occupant_tolerance,
    occupant_tol_file,
    occupant_background,  # TODO: haven't been used just yet
    occupant_preference,
    occupant_pcm,
):
    """
    Create occupant objects based on real dataset and separates them in
    train and test occupants

    Parameters
    ----------
    num_new_occupants: int
        Number of occupants that start the day
    occupant_tolerance: float
        Number in [0,1] that determines how tolerant the occupant its to the
        environment. This number is then used to sample from an occupant
        distribution. A number of -1 means using only real occupants.
    occupant_tol_file: str
        Occupant's tolerance csv path
    occupant_background: str
        Occupants' background csv path
    occupant_preference: str
        Occupants' thermal preference csv path
    occupant_pcm: str
        Occupants' PCM csv path

    Returns
    -------
    dict_occ
        Dictionary with all occupants' objects as values and `user_id` as key.
    df_occ
        Dataframe with historical data of the real occupants.
    """
    # generating synthetic occupants based on real ones from dataset
    # find and load occupants within the tolerance threshold
    dict_pcm, dict_occ, dict_occ_map = load_variable(occupant_pcm), {}, {}
    df_occ_tol = pd.read_csv(occupant_tol_file)

    df_filtered_occupants = df_occ_tol[df_occ_tol["tolerance"] <= occupant_tolerance * 100]
    list_real_occ = list(df_filtered_occupants["user_id"])

    # using exact replicas of occupants from dataset
    if occupant_tolerance == -1:
        list_all_occ = list(pd.read_csv(occupant_background)["user_id"])
        assert len(list_all_occ) >= num_new_occupants
        list_occ = random.sample(list_all_occ, num_new_occupants)

        # initialize new occupant objects
        for occupant in list_occ:
            dict_occ[occupant] = Occupant(occupant, dict_pcm[occupant])

        # historical data for all current occupants
        df_occ = pd.read_csv(occupant_preference)
        df_occ = df_occ[df_occ["user_id"].isin(list_occ)]

    # initialize new occupants objects
    else:
        df_occ = pd.read_csv(occupant_preference)
        df_occ = df_occ[df_occ["user_id"].isin(list_real_occ)]

        # individually create occupants with some real occupant's PCM
        for occupant in range(1, num_new_occupants + 1):
            user_id = f"user_{occupant}"
            real_occ = random.sample(list_real_occ, 1)[0]
            # keep track of real occ used, e.g., {user_1 : dorn2}
            dict_occ_map[user_id] = real_occ
            dict_occ[user_id] = Occupant(user_id, dict_pcm[real_occ])

    return dict_occ, df_occ, dict_occ_map


def zone_loader(
    data_path,
    zone_attributes_file,
    zone_state_action_file,
    weather_file
):
    """
    Load information about the different zones.

    Parameters
    ----------
    data_path: str
        Use case folder path
    zone_attributes_file: str
        JSON file with the zones' attributes
    zone_state_action_file: str
        JSON file name with zones' state and action space
    weather_file: str
        Weather file name

    Returns
    -------
    zones
        Dictonary with all zones' objects with `zone_id` as key.
    observation_spaces
        List of observation space for all zones
    action_spaces
        List of action space for all zones
   """

    with open(zone_state_action_file) as json_file:
        zone_state_action = json.load(json_file)

    with open(zone_attributes_file) as json_file:
        zone_attributes = json.load(json_file)

    zones, observation_spaces, action_spaces = {}, [], []

    # Initialize zone objects based on zone state actions file
    for uid in zone_state_action.keys():
        attributes = zone_attributes[uid]
        # zone object
        zone = Zone(
            zone_id=uid,
            model_type=attributes["model_type"],
            model_features=attributes["model_features"],
            max_num_occupants=attributes["max_num_occupants"],
        )

        # load zone-specific indoor and weather data file
        data_file = str(uid) + ".csv"
        indoor_data = data_path / data_file
        with open(indoor_data) as csv_file:
            indoor_data = pd.read_csv(csv_file)

        with open(weather_file) as csv_file:
            weather_data = pd.read_csv(csv_file)

        for feature, value in zone_state_action[uid]["states"].items():
            if value:
                if "out" in feature:
                    zone.data[feature] = list(weather_data[feature])
                else:
                    zone.data[feature] = list(indoor_data[feature])

        # data-driven model for the zone
        zone.train_model()

        observation_spaces.append(zone.observation_space)
        action_spaces.append(zone.action_space)

        # zones = {uid: zone object}
        zone.reset()
        zones[uid] = zone

    return zones, observation_spaces, action_spaces


class ComfortLearn(gym.Env):
    def __init__(
        self,
        experiment_name,
        seed,
        data_path,
        num_new_occupants,
        zone_selection,
        occupant_timing,
        occupant_tolerance,
        occupant_tol_file,
        occupant_preference,
        occupant_background,
        occupant_pcm,
        zone_attributes,
        weather_file,
        zones_states_actions,
        simulation_period=(0, 23520 - 1),  # every 15min
        cost_function=["unc"],
        central_agent=True,  # TODO: not define on this version
        verbose=True,
    ):
        self.folder_str = experiment_name
        self.seed = self.set_seed(seed)
        print(f"Current seed: {self.seed}")

        # placeholder init
        self.curr_day = None
        self.next_day = None
        self.state = None

        # folder for experiment results
        try:
            os.mkdir(self.folder_str)
        except OSError:
            pass

        # logging file
        logging.basicConfig(
            filename=self.folder_str + "/" + self.folder_str + ".log",
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s",
        )

        # load parameters
        with open(zones_states_actions) as json_file:
            self.zones_states_actions = json.load(json_file)

        # create occupants objects and load their data
        msg = f"Creating occupants based on data from {data_path} ..."

        logging.info(msg)
        if verbose:
            print(msg)

        params_occupant = {
            "num_new_occupants": num_new_occupants,
            "occupant_tolerance": occupant_tolerance,
            "occupant_tol_file": data_path / occupant_tol_file,
            "occupant_background": data_path / occupant_background,
            "occupant_preference": data_path / occupant_preference,
            "occupant_pcm": data_path / occupant_pcm,
        }

        (
            self.dict_occupants,
            self.df_occupants,
            self.dict_occ_map
        ) = occupant_loader(**params_occupant)

        # create thermal preference distribution for human agents and store it
        self.tol = occupant_tolerance
        self.kde = tp_dist(self.df_occupants, self.tol, self.folder_str)

        # TODO: not be hardcoded for future
        self.df_bestzones = pd.read_csv("data/dorn/bestzone_map.csv")

        # generate occupant entering and leaving timings
        self.occ_timing = occupant_timing
        if self.occ_timing == "fixed":
            # enter = 9am, leave = 5pm
            self.enter_time = 9
            self.leave_time = 17
        elif self.occ_timing == "stochastic":
            # randomly sample with standard deviation 2
            # enter = mean of 9am, leaving mean of 5pm
            self.enter_time = np.random.normal(9, 2)
            while self.enter_time <= 7.0:
                # make sure it's above 7am
                self.enter_time = np.random.normal(9, 2)
            self.leave_time = np.random.normal(17, 2)
        else:
            print(f"`occupant_timing` only supports `fixed` or `stochastic` and you type{self.occ_timing})")

        # create zone objects and load their data
        msg = "Creating zones ..."
        logging.info(msg)
        if verbose:
            print(msg)

        params_loader = {
            "data_path": data_path,
            "zone_attributes_file": data_path / zone_attributes,
            "zone_state_action_file": zones_states_actions,
            "weather_file": data_path / weather_file,
        }
        (
            self.zones,
            self.observation_spaces,
            self.action_spaces,
        ) = zone_loader(**params_loader)

        # UNC for real labels
        self.zones_unc = {}  # unc per zone per occupant, dict of dict
        self.zones_unc_avg = {}  # average unc per zone
        self.zones_unc_ts = {}

        # UNC for comfort bands

        # for set-temperature of 26.5
        self.lower_band1 = 25.5
        self.upper_band1 = 27.0
        self.zones_unc_band1 = {}
        self.zones_unc_avg_band1 = {}
        self.zones_unc_ts_band1 = {}

        # for set-temperature of 25
        self.lower_band2 = 24
        self.upper_band2 = 26
        self.zones_unc_band2 = {}
        self.zones_unc_avg_band2 = {}
        self.zones_unc_ts_band2 = {}

        for uid, _ in self.zones.items():
            self.zones_unc_ts[uid] = defaultdict(list)
            self.zones_unc_ts_band1[uid] = defaultdict(list)
            self.zones_unc_ts_band2[uid] = defaultdict(list)

        self.simulation_period = simulation_period
        self.cost_function = cost_function
        self.verbose = verbose
        self.n_zones = len(list(self.zones))
        self.zone_selection = zone_selection

        # initial assignment
        self.assign_zones(self.zone_selection)

        self.reset()

        msg = "Environment created!"
        logging.info(msg)
        if verbose:
            print(msg)

    def get_state_action_spaces(self):
        """Returns state-action spaces for all zones"""
        return self.observation_spaces, self.action_spaces

    def next_time_step(self):
        """Advances simulation to the next time-step"""
        self.time_step = next(self.min_15)
        for zone in self.zones.values():
            zone.time_step = self.time_step

    def step(self, actions):
        s = []  # list of states
        occ_reassigned = False

        for uid, zone in self.zones.items():
            # move occupants between zones when there are more than 1 zone
            # and it's a new workday. Only do it once for all zones
            if (
                self.curr_day != self.next_day
                and len(self.zones.items()) != 1
                and zone.data["day"][self.time_step] not in [5, 6]
                and not occ_reassigned
            ):
                self.assign_zones(self.zone_selection)
                occ_reassigned = True
                if self.verbose:
                    print("Occupants were reassigned to zones!")

            # sampling enter and leave time only if it's a new workday
            if (
                self.occ_timing == "stochastic"
                and self.curr_day != self.next_day
                and zone.data["day"][self.time_step] not in [5, 6]
            ):
                # randomly sample with standard deviation 2
                # enter = mean of 9am, leaving mean of 5pm
                self.enter_time = np.random.normal(8, 2)
                while self.enter_time <= 7.0:
                    # make sure it's above 7am
                    self.enter_time = np.random.normal(8, 2)
                self.leave_time = np.random.normal(17, 2)

                if self.verbose:
                    print("New day timings")
                    print(self.enter_time)
                    print(self.leave_time)

            if self.verbose:
                print(f"Zone: {uid}")
                print(f"Actions to take: {actions}")
                print(f"Current states: {self._get_ob()}")
                print(f"Current occupants: {zone.occupants.keys()}")

            # take actions
            for state_name, value in self.zones_states_actions[uid]["states"].items():
                if actions is None:
                    # no actions are taken, just go through operational data
                    if value:
                        s.append(zone.data[state_name][self.time_step])

                else:
                    # TODO: actually take actions from controller
                    pass

            # calculate new states
            # TODO

            # during 8-17 working hours and workday, get BMS comfort band
            if (
                zone.data["hour"][self.time_step] >= 8
                and zone.data["hour"][self.time_step] <= 17
                and zone.data["day"][self.time_step] not in [5, 6]
            ):
                # for each current occupant on current zone
                for user_id, occupant in zone.occupants.items():
                    # the same occupants won't be in other zones
                    # so their comfort value should be NaN
                    self.fill_unc(uid, user_id, bands=True)

                (
                    self.zones_unc_band1[uid],
                    self.zones_unc_avg_band1[uid],
                    self.zones_unc_ts_band1[uid]
                ) = zone.unc(
                    self.zones_unc_ts_band1[uid].copy(),
                    band=True,
                    lower_temp=self.lower_band1,
                    upper_temp=self.upper_band1
                )

                (
                    self.zones_unc_band2[uid],
                    self.zones_unc_avg_band2[uid],
                    self.zones_unc_ts_band2[uid]
                ) = zone.unc(
                    self.zones_unc_ts_band2[uid].copy(),
                    band=True,
                    lower_temp=self.lower_band2,
                    upper_temp=self.upper_band2
                )
            # outside weekday and workday hours, insert empty values for
            # UNC time-series so that the length matches the simulation period
            elif (
                zone.data["hour"][self.time_step] < 8
                or zone.data["hour"][self.time_step] > 17
                or zone.data["day"][self.time_step] in [5, 6]
            ):
                # for each current occupant on current zone
                for user_id, _ in zone.occupants.items():
                    self.zones_unc_ts_band1[uid][user_id].append(np.nan)
                    self.zones_unc_ts_band2[uid][user_id].append(np.nan)
                    self.fill_unc(uid, user_id, bands=True)

            # during working hours and workday, get thermal preference label
            if (
                zone.data["hour"][self.time_step] >= self.enter_time
                and zone.data["hour"][self.time_step] <= self.leave_time
                and zone.data["day"][self.time_step] not in [5, 6]
            ):
                # for each current occupant on current zone
                for user_id, occupant in zone.occupants.items():
                    # current indoor temp
                    t_in = zone.data["t_in"][self.time_step]

                    # get ground truth termal preference label
                    tp_gt = self.get_tp_gt(t_in)

                    # use the real ground truth as `predicted` one
                    occupant.tp_pred[uid].append(tp_gt)
                    occupant.t_in[uid].append(t_in)

                    # the same occupants won't have labels in other zones
                    # so their UNC time-series value should be NaN
                    self.fill_unc(uid, user_id, bands=False)

                # calculate UNC for current zone and occupants inside
                (
                    self.zones_unc[uid],
                    self.zones_unc_avg[uid],
                    self.zones_unc_ts[uid]
                ) = zone.unc(self.zones_unc_ts[uid].copy())

            # outside weekday and workday hours, insert empty values for
            # UNC time-series so that the length matches the simulation period
            elif (
                zone.data["hour"][self.time_step] < self.enter_time
                or zone.data["hour"][self.time_step] > self.leave_time
                or zone.data["day"][self.time_step] in [5, 6]
            ):
                # for each current occupant on current zone
                for user_id, _ in zone.occupants.items():
                    self.zones_unc_ts[uid][user_id].append(np.nan)
                    self.fill_unc(uid, user_id, bands=False)

            rewards = 0  # TODO: no control just yet
            self.cumulated_reward_episode += rewards
            self.state = np.array(s)  # states are appended just as a list

        # end of zones loop
        if self.verbose:
            print(f"New states: {self._get_ob()}")

        # update current day, advance time step, and get next day
        self.curr_day = self.next_day
        self.next_time_step()
        for uid, zone in self.zones.items():
            # only one zone is needed for this update
            if self.time_step < len(zone.data["day"]):  # don't overflow
                self.next_day = zone.data["day"][self.time_step]
                break

        return self._get_ob(), rewards, self._terminal()

    def reset(self):
        """Variables initialization"""
        self.min_15 = iter(
            np.array(range(self.simulation_period[0], self.simulation_period[1] + 1))
        )
        self.next_time_step()

        self.cumulated_reward_episode = 0
        self.zones_unc = {}
        self.zones_unc_avg = {}
        s = []

        for zone_id, zone in self.zones.items():
            zone.reset()
            for state_name, value in self.zones_states_actions[zone_id][
                "states"
            ].items():
                if value:
                    s.append(zone.data[state_name][self.time_step])

            # placeholder initialization
            self.curr_day = zone.data["day"][self.time_step]
            self.next_day = zone.data["day"][self.time_step]

        self.state = np.array(s)

        return self._get_ob()

    def set_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def assign_zones(self, criterion):
        """
        Assign existing occupants to the available zones

        Parameters
        ----------
        criterion: str
            Way in which occupants will be assigned to the available zones
        """
        # clear existing occupants for both zones
        for _, zones in self.zones.items():
            zones.occupants = {}

        if criterion == "random":
            # randomly assigned users to zones
            for occupant_id, occupant in self.dict_occupants.items():
                curr_zone = random.sample(self.zones_states_actions.keys(), 1)[0]
                self.zones[curr_zone].occupants[occupant_id] = occupant
                if self.verbose:
                    print(f"Occupant {occupant_id} going to {curr_zone}")

        elif criterion == "forced":
            # assuming `perfect information`, use an auxuliary file where each
            # user has a `best zone` in UNC terms
            for occupant_id, occupant in self.dict_occupants.items():
                # get real occupant id
                if "user" in occupant_id:
                    real_id = self.dict_occ_map[occupant_id]
                else:
                    real_id = occupant_id
                curr_zone = self.df_bestzones[self.df_bestzones["user_id"] == real_id]["best_zone"].values[0]

                print(real_id)
                print(curr_zone)

                self.zones[curr_zone].occupants[occupant_id] = occupant
                if self.verbose:
                    print(f"Occupant {occupant_id} going to {curr_zone}")

    def fill_unc(self, curr_zone_uid, user_id, bands=False):
        """
        Occupants in other zones (not the current) will have NaNs as UNC
        values. This is needed for the UNC time-series to be the same length
        as the simulation period
        """
        if self.n_zones > 1:
            # get list of other existing zone uids
            other_zones = list(self.zones.keys())
            other_zones.remove(curr_zone_uid)  # ignore the current zone
            # append NaN for the current user
            for other_uid in other_zones:
                if not bands:
                    self.zones_unc_ts[other_uid][user_id].append(np.nan)
                else:
                    self.zones_unc_ts_band1[other_uid][user_id].append(np.nan)
                    self.zones_unc_ts_band2[other_uid][user_id].append(np.nan)

    def get_tp_gt(self, temp):
        """
        Calculate the thermal preference probabilities given a indoor
        temperature

        Parameters
        ----------
        temp: float
            Indoor temperature at which the occupant is currently exposed

        Returns
        -------
        tp_gt: float
            Thermal preference label
        """
        # get empirical distributions (indoor temp vs label) of occupants
        if self.tol == 0.1:
            # lower values moves the distribution plots.
            # TODO: make this programatically and not hardcoded
            x_c, y_c = self.kde.get_lines()[2].get_data()  # 11.0, cooler
            x_nc, y_nc = self.kde.get_lines()[1].get_data()  # 10.0, no change
            x_w, y_w = self.kde.get_lines()[0].get_data()  # 9.0, prefer warmer
        else:
            x_nc, y_nc = self.kde.get_lines()[2].get_data()  # 10.0, no change
            x_c, y_c = self.kde.get_lines()[1].get_data()  # 11.0, cooler
            x_w, y_w = self.kde.get_lines()[0].get_data()  # 9.0, warmer

        # get label probs at the closest temperature to current temp
        # achieve this by substracting current temp
        x_nc = [abs(x - temp) for x in x_nc]
        x_c = [abs(x - temp) for x in x_c]
        x_w = [abs(x - temp) for x in x_w]

        idx_nc = x_nc.index(min(x_nc))
        idx_c = x_c.index(min(x_c))
        idx_w = x_w.index(min(x_w))

        # get weight per label and normalize them
        nc_label = y_nc[idx_nc]
        c_label = y_c[idx_c]
        w_label = y_w[idx_w]

        normalizer = nc_label + c_label + w_label

        nc_label = nc_label/normalizer
        c_label = c_label/normalizer
        w_label = w_label/normalizer

        # sample label
        tp_gt = choice([9.0, 10.0, 11.0], 1, p=[w_label, nc_label, c_label])

        if self.verbose:
            print(f"Label probs are: No change {nc_label}, Cooler {c_label}, Warmer {w_label}")
            print(f"Chosen label: {tp_gt}")

        return tp_gt[0]  # return array element

    def _get_ob(self):
        return self.state

    def _terminal(self):
        is_terminal = bool(self.time_step >= self.simulation_period[1])
        if is_terminal:
            for zone in self.zones.values():
                zone.terminate()

            # TODO When the simulation is over, convert all the control
            # variables to numpy arrays so they are easier to plot

            # save variables
            for zone_str, zone in self.zones.items():
                save_variable(self.folder_str + "/" + zone_str + "_" + str(self.seed) + ".pkl", zone)

            variables = {
                "cumulated_reward": self.cumulated_reward_episode,
                "dict_zones_unc": self.zones_unc,
                "dict_zones_unc_avg": self.zones_unc_avg,
                "dict_zones_unc_ts": self.zones_unc_ts,
                "dict_zones_unc_band1": self.zones_unc_band1,
                "dict_zones_unc_avg_band1": self.zones_unc_avg_band1,
                "dict_zones_unc_ts_band1": self.zones_unc_ts_band1,
                "dict_zones_unc_band2": self.zones_unc_band2,
                "dict_zones_unc_avg_band2": self.zones_unc_avg_band2,
                "dict_zones_unc_ts_band2": self.zones_unc_ts_band2,
            }

            for name, var in variables.items():
                save_variable(self.folder_str + "/" + name + "_" + str(self.seed) + ".pkl", var)

            if self.verbose:
                msg = f"Cumulated reward: {str(self.cumulated_reward_episode)}"
                logging.info(msg)
                print(msg)

        return is_terminal
