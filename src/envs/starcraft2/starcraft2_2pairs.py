from ..multiagentenv import MultiAgentEnv
from .map_params import get_map_params, map_present
from utils.dict2namedtuple import convert
from operator import attrgetter
from copy import deepcopy
from absl import flags
import numpy as np
import pygame
import sys
import os
import math

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

FLAGS = flags.FLAGS
FLAGS(['main.py'])

_possible_results = {
    sc_pb.Victory: 1,
    sc_pb.Defeat: -1,
    sc_pb.Tie: 0,
    sc_pb.Undecided: 0,
}

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

action_move_id = 16     #    target: PointOrUnit
action_attack_id = 23   #    target: PointOrUnit
action_stop_id = 4      #    target: None
action_heal_id = 386    #    target: Unit

'''
StarCraft II
'''

class SC2(MultiAgentEnv):

    def __init__(self, **kwargs):
        self.glob_args = kwargs["args"]

        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)
        # Read arguments
        self.map_name = args.map_name
        assert map_present(self.map_name), \
            "map {} not in map registry! please add.".format(self.map_name)
        map_params = convert(get_map_params(self.map_name))
        self.n_agents = map_params.n_agents
        self.n_enemies = map_params.n_enemies
        self.episode_limit = map_params.limit
        self._move_amount = args.move_amount
        self._step_mul = args.step_mul
        self.difficulty = args.difficulty
        # Observations and state
        self.obs_own_health = args.obs_own_health
        self.obs_all_health = args.obs_all_health
        self.obs_instead_of_state = args.obs_instead_of_state
        self.obs_last_action = args.obs_last_action
        self.obs_pathing_grid = args.obs_pathing_grid
        self.obs_terrain_height = args.obs_terrain_height
        self.state_last_action = args.state_last_action
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9
        # Rewards args
        self.reward_sparse = args.reward_sparse
        self.reward_only_positive = args.reward_only_positive
        self.reward_negative_scale = args.reward_negative_scale
        self.reward_death_value = args.reward_death_value
        self.reward_win = args.reward_win
        self.reward_defeat = args.reward_defeat
        self.reward_scale = args.reward_scale
        self.reward_scale_rate = args.reward_scale_rate
        # Other
        self.continuing_episode = args.continuing_episode
        self.seed = args.seed
        self.heuristic = args.heuristic
        self.window_size = (1920, 1200)
        self.save_replay_prefix = args.save_replay_prefix
        self.restrict_actions = args.restrict_actions

        # For sanity check
        self.debug_inputs = False # DEBUG
        self.debug_rewards = False

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params.a_race
        self._bot_race = map_params.b_race
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params.unit_type_bits
        self.map_type = map_params.map_type

        # MACKRL arguments
        self.mackrl_delegate_if_zero_ck = getattr(kwargs["args"], "mackrl_delegate_if_zero_ck", False)
        self.relax_pairwise_aa = args.relax_pairwise_aa if hasattr(args, "relax_pairwise_aa") else False

        if sys.platform == 'linux':
            os.environ['SC2PATH'] = os.path.join(os.getcwd(), "3rdparty", 'StarCraftII')
            self.game_version = args.game_version
        else:
            # Can be derived automatically
            self.game_version = None

        # Launch the game
        self._launch()

        self.max_reward = self.n_enemies * self.reward_death_value + self.reward_win
        self._game_info = self.controller.game_info()
        self._map_info = self._game_info.start_raw
        self.map_x = self._map_info.map_size.x
        self.map_y = self._map_info.map_size.y
        self.map_play_area_min = self._map_info.playable_area.p0
        self.map_play_area_max = self._map_info.playable_area.p1
        self.max_distance_x = self.map_play_area_max.x - self.map_play_area_min.x
        self.max_distance_y = self.map_play_area_max.y - self.map_play_area_min.y
        self.terrain_height = np.flip(np.transpose(np.array(list(self._map_info.terrain_height.data)).reshape(self.map_x, self.map_y)), 1)
        self.pathing_grid = np.flip(np.transpose(np.array(list(self._map_info.pathing_grid.data)).reshape(self.map_x, self.map_y)), 1)

        self._episode_count = 0
        self._total_steps = 0

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0

        self.last_stats = None

        self.obs_noise = getattr(self.glob_args, "obs_noise", False)
        self.obs_noise_std = getattr(self.glob_args, "obs_noise_std", False)

        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0

        self.stalker_id = self.colossus_id = self.zealot_id = 0

    def init_ally_unit_types(self, min_unit_type):
        # This should be called once from the init_units function

        self.stalker_id = self.sentry_id = self.zealot_id = self.colossus_id = 0
        self.marine_id = self.marauder_id= self.medivac_id = 0
        self.hydralisk_id = 0

        self._min_unit_type = min_unit_type
        if self.map_type == "marines":
            self.marine_id = min_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "colossi_and_zealots":
            self.colossus_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "hydralisks_and_zerglings":
            self.hydralisk_id = min_unit_type
            self.zergling_id = min_unit_type + 1
        elif self.map_type == "stalkers":
            self.hydralisk_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "MMM":
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = min_unit_type
        elif self.map_type == "marauders":
            self.marauders_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = min_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = min_unit_type
        elif self.map_type == "bane":
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1

        # if self.map_type == 'sz' or self.map_type == 's_v_z' or self.map_type == "stalkers_and_zealots":
        #     self.stalker_id = min_unit_type
        #     self.zealot_id = min_unit_type + 1
        # elif self.map_type == 'MMM':
        #     self.marauder_id = min_unit_type
        #     self.marine_id = min_unit_type + 1
        #     self.medivac_id = min_unit_type + 2
        # elif self.map_type == 'zealots':
        #     self.zealot_id = min_unit_type
        # elif self.map_type == 'focus_fire':
        #     self.hydralisk_id = min_unit_type
        # elif self.map_type == 'retarget':
        #     self.stalker_id = min_unit_type
        # elif self.map_type == 'colossus':
        #     self.colossus_id = min_unit_type

    def _launch(self):

        self._run_config = run_configs.get()
        self._map = maps.get(self.map_name)

        # Setting up the interface
        self.interface = sc_pb.InterfaceOptions(
                raw = True, # raw, feature-level data
                score = True)

        self._sc2_proc = self._run_config.start(game_version=self.game_version, window_size=self.window_size)
        self.controller = self._sc2_proc.controller

        # Create the game.
        create = sc_pb.RequestCreateGame(realtime = False,
                random_seed = self.seed,
                local_map=sc_pb.LocalMap(map_path=self._map.path, map_data=self._run_config.map_data(self._map.path)))
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties[self.difficulty])
        self.controller.create_game(create)

        join = sc_pb.RequestJoinGame(race=races[self._agent_race], options=self.interface)
        self.controller.join_game(join)

    def save_replay(self):
        prefix = self.save_replay_prefix or self.map_name
        replay_path = self._run_config.save_replay(self.controller.save_replay(), replay_dir='', prefix=prefix)
        print("Replay saved at: %s" % replay_path)

    def reset(self, obs_noise_std=None):
        """Start a new episode."""

        if self.debug_inputs or self.debug_rewards:
            print('------------>> RESET <<------------')

        self._episode_steps = 0
        if self._episode_count > 0:
            # No need to restart for the first episode.
            self._restart()

        self._episode_count += 1

        if self.heuristic:
            self.heuristic_targets = [0] * self.n_agents

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_agent_units = None
        self.previous_enemy_units = None

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        try:
            self._obs = self.controller.observe()
            self.init_units()
        except protocol.ProtocolError:
            self.full_restart()
        except protocol.ConnectionError:
            self.full_restart()

        #print(self.controller.query(q_pb.RequestQuery(abilities=[q_pb.RequestQueryAvailableAbilities(unit_tag=self.agents[0].tag)])))
        #print(self.controller.data_raw())

        if self.obs_noise:
            self._noise_obs(obs_noise_std=obs_noise_std if obs_noise_std is not None else 0.0)

        return self.get_obs(), self.get_state()

    def _restart(self):

        # Kill and restore all units
        try:
            self.kill_all_units()
            self.controller.step(2)
        except protocol.ProtocolError:
            self.full_restart()
        except protocol.ConnectionError:
            self.full_restart()

    def full_restart(self):
        # End episode and restart a new one
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def one_hot(self, data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def step(self, actions, obs_noise_std=None):

        actions = [int(a) for a in actions]

        self.last_action = self.one_hot(actions, self.n_actions)

        # Collect individual actions
        sc_actions = []
        for a_id, action in enumerate(actions):
            if not self.heuristic:
                agent_action = self.get_agent_action(a_id, action)
            else:
                agent_action = self.get_agent_action_heuristic(a_id, action)
            if agent_action:
              sc_actions.append(agent_action)
        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)

        try:
            res_actions = self.controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self.controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self.controller.observe()
        except protocol.ProtocolError:
            self.full_restart()
            return 0, True, {}
        except protocol.ConnectionError:
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update what we know about units
        end_game = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        if end_game is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if end_game == 1:
                self.battles_won += 1
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1

            elif end_game == -1:
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self.episode_limit > 0 and self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug_inputs or self.debug_rewards:
            print("Total Reward = %.f \n ---------------------" % reward)

        if self.reward_scale:
            reward /= (self.max_reward / self.reward_scale_rate)

        if self.obs_noise:
            self._noise_obs(obs_noise_std=obs_noise_std if obs_noise_std is not None else 0.0) # create noised observations data

        return reward, terminated, info

    def _noise_obs(self, obs_noise_std):
        class Obj: pass
        self.agents_noisy = {}
        for ak, _ in self.agents.items():
            self.agents_noisy[ak] = {}
            for ak2, agent in self.agents.items():
                self.agents_noisy[ak][ak2] = Obj()
                self.agents_noisy[ak][ak2].pos = Obj()
                self.agents_noisy[ak][ak2].pos.x = agent.pos.x + np.random.normal(0.0, obs_noise_std)
                self.agents_noisy[ak][ak2].pos.y = agent.pos.y + np.random.normal(0.0, obs_noise_std)
                self.agents_noisy[ak][ak2].health = agent.health
                self.agents_noisy[ak][ak2].health_max = agent.health_max
                self.agents_noisy[ak][ak2].unit_type = agent.unit_type
                self.agents_noisy[ak][ak2].shield = agent.shield

        self.enemies_noisy = {}
        for ak, _ in self.agents.items():
            self.enemies_noisy[ak] = {}
            for ek, enemy in self.enemies.items():
                self.enemies_noisy[ak][ek] = Obj()
                self.enemies_noisy[ak][ek].pos = Obj()
                self.enemies_noisy[ak][ek].pos.x = enemy.pos.x + np.random.normal(0.0, obs_noise_std)
                self.enemies_noisy[ak][ek].pos.y = enemy.pos.y + np.random.normal(0.0, obs_noise_std)
                self.enemies_noisy[ak][ek].health = enemy.health
                self.enemies_noisy[ak][ek].health_max = enemy.health_max
                self.enemies_noisy[ak][ek].unit_type = enemy.unit_type
                self.enemies_noisy[ak][ek].shield = enemy.shield
        pass

    def get_agent_action(self, a_id, action):

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        true_avail_actions = self.get_avail_agent_actions(a_id)
        if true_avail_actions[action] == 0:
            action = 1

        if action == 0:
            # no-op (valid only when dead)
            try:
                assert unit.health == 0, "No-op chosen but the agent's unit is not dead"
            except Exception as e:
                pass
            if self.debug_inputs:
                print("Agent %d: Dead"% a_id)
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_stop_id,
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: Stop"% a_id)

        elif action == 2:
            # north
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x, y = y + self._move_amount),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: North"% a_id)

        elif action == 3:
            # south
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x, y = y - self._move_amount),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: South"% a_id)

        elif action == 4:
            # east
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x + self._move_amount, y = y),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: East"% a_id)

        elif action == 5:
            # west
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x - self._move_amount, y = y),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: West"% a_id)
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == 'MMM' and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_id = action_heal_id
            else:
                target_unit = self.enemies[target_id]
                action_id = action_attack_id
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(ability_id = action_id,
                    target_unit_tag = target_tag,
                    unit_tags = [tag],
                    queue_command = False)

            if self.debug_inputs:
                print("Agent %d attacks enemy # %d" % (a_id, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target_tag = self.enemies[self.heuristic_targets[a_id]].tag
        action_id = action_attack_id

        cmd = r_pb.ActionRawUnitCommand(ability_id = action_id,
                target_unit_tag = target_tag,
                unit_tags = [tag],
                queue_command = False)

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def reward_battle(self):

        if self.reward_sparse:
            return 0

        #  delta health - delta enemies + delta deaths where value:
        #   if enemy unit dies, add reward_death_value per dead unit
        #   if own unit dies, subtract reward_death_value per dead unit

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        if self.debug_rewards:
            for al_id in range(self.n_agents):
                print("Agent %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_agent_units[al_id].health - self.agents[al_id].health, self.previous_agent_units[al_id].shield - self.agents[al_id].shield))
            print('---------------------')
            for al_id in range(self.n_enemies):
                print("Enemy %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_enemy_units[al_id].health - self.enemies[al_id].health, self.previous_enemy_units[al_id].shield - self.enemies[al_id].shield))

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = self.previous_agent_units[al_id].health + self.previous_agent_units[al_id].shield
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += (prev_health - al_unit.health - al_unit.shield) * neg_scale

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = self.previous_enemy_units[e_id].health + self.previous_enemy_units[e_id].shield
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:

            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Reward: ", delta_enemy + delta_deaths)
                print("--------------------------")

            reward = delta_enemy + delta_deaths
            reward = abs(reward) # shield regeration
        else:
            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Delta ally: ", - delta_ally)
                print("Reward: ", delta_enemy + delta_deaths)
                print("--------------------------")

            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        return self.n_actions

    def distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        return 6

    def unit_sight_range(self, agent_id):
        return 9

    def unit_max_cooldown(self, unit):

        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1
        }
        return switcher.get(unit.unit_type, 15)

        # if self.map_type in ['marines', 'm_z']:
        #     return 15
        #
        # unit = self.get_unit_by_id(agent_id)
        # if unit.unit_type == self.marine_id:
        #     return 15
        # if unit.unit_type == self.marauder_id:
        #     return 25
        # if unit.unit_type == self.medivac_id:
        #     return 200
        #
        # if unit.unit_type == self.stalker_id:
        #     return 35
        # if unit.unit_type == self.zealot_id:
        #     return 22
        # if unit.unit_type == self.colossus_id:
        #     return 24
        # if unit.unit_type == self.sentry_id:
        #     return 13
        # if unit.unit_type == self.hydralisk_id:
        #     return 10

    def unit_max_shield(self, unit):

        if unit.unit_type == 74 or unit.unit_type == self.stalker_id: # Protoss's Stalker
            return 80
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id: # Protoss's Zaelot
            return 50
        if unit.unit_type == 77 or unit.unit_type == self.sentry_id: # Protoss's Sentry
            return 40
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id: # Protoss's Colossus
            return 150
        else:
            return -1

    def can_move(self, unit, direction):

        m = self._move_amount / 2

        if direction == 0: # north
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == 1: # south
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == 2: # east
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else : # west
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        try:
            if self.pathing_grid[x, y] == 0:
                return True
        except Exception as e:
            return False

        return False

    def circle_grid_coords(self, unit, radius):
        # Generates coordinates in grid that lie within a circle

        r = radius
        x_floor = int(unit.pos.x)
        y_floor = int(unit.pos.y)

        points = []
        for x in range(-r, r + 1):
            Y = int(math.sqrt(abs(r*r-x*x))) # bound for y given x
            for y in range(- Y, + Y + 1):
                points.append((x_floor + x, y_floor + y))
        return points

    def get_surrounding_points(self, unit, include_self=False):

        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma), (x, y - 2 * ma),
            (x + 2 * ma, y), (x - 2 * ma, y),
            (x + ma, y + ma), (x - ma, y - ma),
            (x + ma, y - ma), (x - ma, y + ma)
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        return x >= 0 and y >=0 and x < self.map_x and y < self.map_y

    def get_surrounding_pathing(self, unit):

        points = self.get_surrounding_points(unit, include_self=False)
        vals = [self.pathing_grid[x, y] / 255 if self.check_bounds(x, y) else 1 for x, y in points]
        return vals

    def get_surrounding_height(self, unit):

        points = self.get_surrounding_points(unit, include_self=True)
        vals = [self.terrain_height[x, y] / 255 if self.check_bounds(x, y) else 1 for x, y in points]
        return vals

    def get_obs_agent(self, agent_id, obs_noise=False):
        # unit: replace with noisy unit

        unit = self.get_unit_by_id(agent_id)

        nf_al = 4 + self.unit_type_bits
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
            nf_en += 1 + self.shield_bits_enemy

        if self.obs_last_action:
            nf_al += self.n_actions

        nf_own = self.unit_type_bits
        if self.obs_own_health:
            nf_own += 1 + self.shield_bits_ally

        move_feats_len = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats_len += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats_len += self.n_obs_height

        move_feats = np.zeros(move_feats_len, dtype=np.float32) # exclude no-op & stop
        enemy_feats = np.zeros((self.n_enemies, nf_en), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, nf_al), dtype=np.float32)
        own_feats = np.zeros(nf_own, dtype=np.float32)

        if unit.health > 0: # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[ind: ind + self.n_obs_pathing] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            if obs_noise:
                enemies = self.enemies_noisy[agent_id]
            else:
                enemies = self.enemies

            for e_id, e_unit in enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if dist < sight_range and e_unit.health > 0: # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id] # available
                    enemy_feats[e_id, 1] = dist / sight_range # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = e_unit.health / e_unit.health_max # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            if max_shield != -1:
                                enemy_feats[e_id, ind] = e_unit.shield / max_shield # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1 # unit type

            # Ally features
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):

                if obs_noise:
                    al_unit = self.get_unit_by_id_noisy(obs_id=agent_id, get_id=al_id)
                else:
                    al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if dist < sight_range and al_unit.health > 0: # visible and alive
                    ally_feats[i, 0] = 1 # visible
                    ally_feats[i, 1] = dist / sight_range # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = al_unit.health / al_unit.health_max # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            if max_shield != -1:
                                ally_feats[i, ind] = al_unit.shield / max_shield # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        try:
                            ally_feats[i, ind + type_id] = 1
                        except Exception as e:
                            pass
                        ind += self.unit_type_bits

                    if self.obs_last_action:
                        ally_feats[i, ind:] = self.last_action[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1

        agent_obs = np.concatenate((move_feats.flatten(),
                                    enemy_feats.flatten(),
                                    ally_feats.flatten(),
                                    own_feats.flatten()))

        if self.debug_inputs:
            print("***************************************")
            print("Agent: ", agent_id)
            print("Available Actions\n", self.get_avail_agent_actions(agent_id))
            print("Move feats\n", move_feats)
            print("Enemy feats\n", enemy_feats)
            print("Ally feats\n", ally_feats)
            print("Own feats\n", own_feats)
            print("***************************************")

        return agent_obs

    def get_obs(self, obs_noise=False):
        agents_obs = [self.get_obs_agent(i, obs_noise=obs_noise) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):

        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)

                ally_state[al_id, 0] = al_unit.health / al_unit.health_max # health
                if self.map_type == 'MMM' and al_unit.unit_type == self.medivac_id:
                    ally_state[al_id, 1] = al_unit.energy / max_cd # energy
                else:
                    ally_state[al_id, 1] = al_unit.weapon_cooldown / max_cd # cooldown
                ally_state[al_id, 2] = (x - center_x) / self.max_distance_x # relative X
                ally_state[al_id, 3] = (y - center_y) / self.max_distance_y # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = al_unit.shield / max_shield # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, ind + type_id] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = e_unit.health / e_unit.health_max # health
                enemy_state[e_id, 1] = (x - center_x) / self.max_distance_x # relative X
                enemy_state[e_id, 2] = (y - center_y) / self.max_distance_y # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = e_unit.shield / max_shield # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        state = state.astype(dtype=np.float32)

        if self.debug_inputs:
            print("------------ STATE ---------------")
            print("Ally state\n", ally_state)
            print("Enemy state\n", enemy_state)
            print("Last action\n", self.last_action)
            print("----------------------------------")

        return state

    def get_unit_type_id(self, unit, ally):

        """Returns the ID of unit type in the given scenario."""

        if ally:  # use new SC2 unit types
            if self.map_type == "colossi_and_zealots":
                if unit.unit_type == 1927:
                    type_id = 0
                else:
                    type_id = 1
            else:
                type_id = unit.unit_type - self._min_unit_type

        else:  # use default SC2 unit types
            if self.map_type == "stalkers_and_zealots":
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            if self.map_type == "marines_and_zealots":
                if unit.unit_type == 73:
                    type_id = 0
                else:
                    type_id = 1
            if self.map_type == "marines_and_zealots_and_stalkers":
                if unit.unit_type == 73:
                    type_id = 0
                if unit.unit_type == 74:
                    type_id = 1
                else:
                    type_id = 2
            if self.map_type == "colossi_and_zealots":
                if unit.unit_type == 73:
                    type_id = 0
                else:
                    type_id = 1
            if self.map_type == "hydralisks_and_zerglings":
                if unit.unit_type == 73:
                    type_id = 0
                else:
                    type_id = 1
            if self.map_type == "bane":
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2

        return type_id
        # if ally: # we use new SC2 unit types
        #
        #     if self.map_type == 'sz':
        #         type_id = unit.unit_type - self.stalker_id
        #     elif self.map_type == 'MMM':
        #         if unit.unit_type == self.marauder_id:
        #             type_id = 0
        #         elif unit.unit_type == self.marine_id:
        #             type_id = 1
        #         else:
        #             type_id = 2
        #
        # else: # 'We use default SC2 unit types'
        #
        #     if self.map_type == 'sz':
        #         # id(Stalker) = 74, id(Zealot) = 73
        #         type_id = unit.unit_type - 73
        #     elif self.map_type == 'MMM':
        #         if unit.unit_type == 51:
        #             type_id = 0
        #         elif unit.unit_type == 48:
        #             type_id = 1
        #         else:
        #             type_id = 2

        #return type_id

    def get_state_size(self):

        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions

        return size

    def get_avail_agent_actions(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot do no-op as alife
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, 0):
                avail_actions[2] = 1
            if self.can_move(unit, 1):
                avail_actions[3] = 1
            if self.can_move(unit, 2):
                avail_actions[4] = 1
            if self.can_move(unit, 3):
                avail_actions[5] = 1

            # can attack only those who is alife
            # and in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if self.map_type == 'MMM' and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves and other flying units
                target_items = [(t_id, t_unit) for (t_id, t_unit) in self.agents.items() if t_unit.unit_type != self.medivac_id]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_unrestricted_actions(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot do no-op as alife
            avail_actions = [1] * self.n_actions
            avail_actions[0] = 0
        else:
            avail_actions = [0] * self.n_actions
            avail_actions[0] = 1
        return avail_actions

    def get_avail_actions(self):

        avail_actions = []
        for agent_id in range(self.n_agents):
            if self.restrict_actions:
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            else:
                avail_agent = self.get_unrestricted_actions(agent_id)
                avail_actions.append(avail_agent)

        return avail_actions

    def get_obs_size(self):

        nf_al = 4 + self.unit_type_bits
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
            nf_en += 1 + self.shield_bits_enemy

        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al

        return move_feats + enemy_feats + ally_feats + own_feats

    def close(self):
        self._sc2_proc.close()

    def render(self):
        pass

    def kill_all_units(self):

        units_alive = [unit.tag for unit in self.agents.values() if unit.health > 0] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        debug_command = [d_pb.DebugCommand(kill_unit = d_pb.DebugKillUnit(tag = units_alive))]
        self.controller.debug(debug_command)

    def init_units(self):

        # In case controller step fails
        while True:

            self.agents = {}
            self.enemies = {}

            ally_units = [unit for unit in self._obs.observation.raw_data.units if unit.owner == 1]
            ally_units_sorted = sorted(ally_units, key=attrgetter('unit_type', 'pos.x', 'pos.y'), reverse=False)

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug_inputs:
                    print("Unit %d is %d, x = %.1f, y = %1.f"  % (len(self.agents), self.agents[i].unit_type, self.agents[i].pos.x, self.agents[i].pos.y))

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 1:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 1:
                min_unit_type = min(unit.unit_type for unit in self.agents.values())
                self.init_ally_unit_types(min_unit_type)

            if len(self.agents) == self.n_agents and len(self.enemies) == self.n_enemies:
                # All good
                return

            try:
                self.controller.step(1)
                self._obs = self.controller.observe()
            except protocol.ProtocolError:
                self.full_restart()
                self.reset()
            except protocol.ConnectionError:
                self.full_restart()
                self.reset()

    def update_units(self):

        # This function assumes that self._obs is up-to-date
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_agent_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated: # means dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated: # means dead
                e_unit.health = 0

        if self.heuristic:
            for al_id, al_unit in self.agents.items():
                current_target = self.heuristic_targets[al_id]
                if current_target == 0 or self.enemies[current_target].health == 0:
                    x = al_unit.pos.x
                    y = al_unit.pos.y
                    min_dist = 32
                    min_id = -1
                    for e_id, e_unit in self.enemies.items():
                        if e_unit.health > 0:
                            dist = self.distance(x, y, e_unit.pos.x, e_unit.pos.y)
                            if dist < min_dist:
                                min_dist = dist
                                min_id = e_id
                    self.heuristic_targets[al_id] = min_id

        if (n_ally_alive == 0 and n_enemy_alive > 0) or self.only_medivac_left(ally=True):
            return -1 # loss
        if (n_ally_alive > 0 and n_enemy_alive == 0) or self.only_medivac_left(ally=False):
            return 1 # win
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0 # tie, not sure if this is possible

        return None

    def only_medivac_left(self, ally):
        if self.map_type != 'MMM':
            return False

        if ally:
            units_alive = [a for a in self.agents.values() if (a.health > 0 and a.unit_type != self.medivac_id)]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [a for a in self.enemies.values() if (a.health > 0 and a.unit_type != self.medivac_id)]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        return self.agents[a_id]

    def get_unit_by_id_noisy(self, obs_id, get_id):
        return self.agents_noisy[obs_id][get_id]

    def get_stats(self):
        stats = {}
        stats["battles_won"] = self.battles_won
        stats["battles_game"] = self.battles_game
        stats["battles_draw"] = self.timeouts
        stats["win_rate"] = self.battles_won / self.battles_game
        stats["timeouts"] = self.timeouts
        stats["restarts"] = self.force_restarts
        return stats

    def get_agg_stats(self, stats):

        current_stats = {}
        for stat in stats:
            for _k, _v in stat.items():
                if not (_k in current_stats):
                    current_stats[_k] = []
                if _k in ["win_rate"]:
                    continue
                current_stats[_k].append(_v)

        # average over stats
        aggregate_stats = {}
        for _k, _v in current_stats.items():
            if _k in ["win_rate"]:
                aggregate_stats[_k] = np.mean([ (_a - _b)/(_c - _d) for _a, _b, _c, _d in zip(current_stats["battles_won"],
                                                                                              [0]*len(current_stats["battles_won"]) if self.last_stats is None else self.last_stats["battles_won"],
                                                                                              current_stats["battles_game"],
                                                                                              [0]*len(current_stats["battles_game"]) if self.last_stats is None else
                                                                                              self.last_stats["battles_game"])
                                                if (_c - _d) != 0.0])
            else:
                aggregate_stats[_k] = np.mean([_a-_b for _a, _b in zip(_v, [0]*len(_v) if self.last_stats is None else self.last_stats[_k])])

        self.last_stats = current_stats
        return aggregate_stats

    # TODO Mika - not sure if we need these anymore and whether they are correct after my changes
    def get_intersect(self, coordinates, e_unit, sight_range ):
        e_x = e_unit.pos.x
        e_y = e_unit.pos.y
        distances = np.sum((coordinates - np.array([e_x, e_y] ))**2, 1)**0.5
        if max( distances ) > sight_range:
            return False
        else:
            return True

    def get_obs_intersection_noisy(self, agent_ids):
        # return array of all noisy obs intersections, by observing agent id (agent_ids)
        return [self.get_obs_intersection(agent_ids, obs_id=_a, obs_noise=True) for _a in agent_ids]

    def get_obs_intersection(self, agent_ids, obs_id=None, obs_noise=False):

        """ Returns the intersection of all of agent_ids agents' observations. """
        # Create grid
        nf_al = 4
        nf_en = 5

        if self.map_name in ['2s_3z', '3s_4z', '4s_5z', '3s_5z']:
            # unit types (in onehot)
            nf_al += 2
            nf_en += 2

        # move_feats = np.zeros(self.n_actions_no_attack - 2, dtype=np.float32) # exclude no-op & stop
        enemy_feats = -1*np.ones((len(agent_ids), self.n_enemies, nf_en), dtype=np.float32)
        ally_feats = -1*np.ones((len(agent_ids), self.n_agents, nf_al), dtype=np.float32)
        #state = np.concatenate((enemy_feats.flatten(),
        #                            ally_feats.flatten()))
        # state = state.astype(dtype=np.float32)
        state = np.concatenate((enemy_feats.flatten(),
                                ally_feats.flatten()))
        state = state.astype(dtype=np.float32)
        #Todo: Check that the dimensions are consistent.
        aa1 = np.array(self.get_avail_agent_actions(agent_ids[0]))
        aa2 = np.array(self.get_avail_agent_actions(agent_ids[1]))
        if self.relax_pairwise_aa: # OR mode
            aa1 = [ 1 if ((aa1[_i] == 1) or (aa2[_i] == 1)) else 0 for _i in range(aa1.shape[0]) ]
            aa2 = aa1.copy()
        a_a1 = np.reshape( aa1, [-1,1])
        a_a2 = np.reshape( aa2, [1,-1])
        avail_actions = a_a1.dot(a_a2)
        # Debug!! TODO: FIX THE IF TRUE BELOW
        if self.mackrl_delegate_if_zero_ck:
            avail_all = avail_actions.copy() * 0
        else:
            avail_all = avail_actions.copy() * 0 + 1

        coordinates = np.zeros([len(agent_ids), 2])
        if obs_noise:
            agents = self.agents_noisy[obs_id]
        else:
            agents = self.agents
        for i, a_id in enumerate(agent_ids):
            if not (agents[a_id].health > 0):
                return state, avail_all * 0.0 # FORCE TO DELEGATE
            else:
                coordinates[i] = [agents[a_id].pos.x, agents[a_id].pos.y]

        # Calculate pairwise distances
        distances = ((coordinates[:, 0:1] - coordinates[:, 0:1].T)**2 + (coordinates[:, 1:2] - coordinates[:, 1:2].T)**2)**0.5
        sight_range = self.unit_sight_range(agent_ids[0])

        # Check that max pairwise distance is less than sight_range.
        if np.max(distances) > sight_range:
            return state, avail_all

        x = np.mean(coordinates, 0)[0]
        y = np.mean(coordinates, 0)[1]

        if obs_noise:
            enemies = self.enemies_noisy[obs_id]
        else:
            enemies = self.enemies
        for e_id, e_unit in enemies.items():
            e_x = e_unit.pos.x
            e_y = e_unit.pos.y
            # dist = self.distance(x, y, e_x, e_y)

            if self.get_intersect(coordinates, e_unit, sight_range) and e_unit.health > 0:  # visible and alive
                # Sight range > shoot range
                for i, a_id in enumerate(agent_ids):
                    dist = self.distance(self.agents[a_id].pos.x, self.agents[a_id].pos.y, e_x, e_y)
                    enemy_feats[i, e_id, 0] = a_a1[self.n_actions_no_attack + e_id, 0]   # available
                    enemy_feats[i, e_id, 1] = dist / sight_range # distance
                    enemy_feats[i, e_id, 2] = (e_x - self.agents[a_id].pos.x) / sight_range # relative X
                    enemy_feats[i, e_id, 3] = (e_y - self.agents[a_id].pos.y) / sight_range # relative Y
                    enemy_feats[i, e_id, 4] = a_a2[0, self.n_actions_no_attack + e_id]  # available

                    if self.map_name in ['2s_3z', '3s_4z', '4s_5z', '3s_5z']:
                        type_id = e_unit.unit_type - 73  # id(Stalker) = 74, id(Zealot) = 73
                        enemy_feats[i, e_id, 4 + type_id] = 1

            else:
                avail_actions[self.n_actions_no_attack + e_id, :] = 0
                avail_actions[:, self.n_actions_no_attack + e_id] = 0

        # place the features of the agent himself always at the first place
        al_ids = list(agent_ids)
        for al in range(self.n_agents):
            if al not in agent_ids:
                al_ids.append(al)
        for i, al_id in enumerate(al_ids):
            if obs_noise:
                al_unit = self.get_unit_by_id_noisy(obs_id=obs_id, get_id=al_id)
            else:
                al_unit = self.get_unit_by_id(al_id)
            al_x = al_unit.pos.x
            al_y = al_unit.pos.y
            #dist = self.distance(x, y, al_x, al_y)
            if self.get_intersect(coordinates, al_unit, sight_range) and al_unit.health > 0:  # visible and alive
                for j, a_id in enumerate(agent_ids):
                    dist = self.distance(agents[a_id].pos.x, agents[a_id].pos.y, al_x, al_y)
                    ally_feats[j, i, 0] = 1  # visible
                    ally_feats[j, i, 1] = dist / sight_range # distance
                    ally_feats[j, i, 2] = (al_x - agents[a_id].pos.x) / sight_range  # relative X
                    ally_feats[j, i, 3] = (al_y - agents[a_id].pos.y) / sight_range  # relative Y

                    if self.map_name in ['2s_3z', '3s_4z', '4s_5z', '3s_5z']:
                        type_id = al_unit.unit_type - self.stalker_id  # id(Stalker) = self.stalker_id, id(Zealot) = self.zealot_id
                        ally_feats[j, i, 4 + type_id] = 1

        state = np.concatenate((enemy_feats.flatten(),
                                ally_feats.flatten()))

        state = state.astype(dtype=np.float32)

        if self.debug_inputs:
            print("***************************************")
            print("Agent_intersections: ", agent_ids)
            print("Enemy feats\n", enemy_feats)
            print("Ally feats\n", ally_feats)
            print("***************************************")
        return state, avail_actions

    def get_obs_intersect_pair_size(self):
        return self.get_obs_intersect_size()*2

    def get_obs_intersect_all_size(self):
        return self.get_obs_intersect_size()*self.n_agents

    def get_obs_intersect_size(self):

        nf_al = 4
        nf_en = 5

        if self.map_name in ['2s_3z', '3s_4z', '4s_5z', '3s_5z']:
            nf_al += 2
            nf_en += 2

        enemy_feats = self.n_enemies *  nf_en
        ally_feats = (self.n_agents) * nf_al

        return  enemy_feats + ally_feats

from components.transforms import _seq_mean

class StatsAggregator():

    def __init__(self):
        self.last_stats = None
        self.stats = []
        pass

    def aggregate(self, stats, add_stat_fn):

        current_stats = {}
        for stat in stats:
            for _k, _v in stat.items():
                if not (_k in current_stats):
                    current_stats[_k] = []
                if _k in ["win_rate"]:
                    continue
                current_stats[_k].append(_v)

        # average over stats
        aggregate_stats = {}
        for _k, _v in current_stats.items():
            if _k in ["win_rate"]:
                aggregate_stats[_k] = np.mean([ (_a - _b)/(_c - _d) for _a, _b, _c, _d in zip(current_stats["battles_won"],
                                                                                              [0]*len(current_stats["battles_won"]) if self.last_stats is None else self.last_stats["battles_won"],
                                                                                              current_stats["battles_game"],
                                                                                              [0]*len(current_stats["battles_game"]) if self.last_stats is None else
                                                                                              self.last_stats["battles_game"])
                                                if (_c - _d) != 0.0])
            else:
                aggregate_stats[_k] = np.mean([_a-_b for _a, _b in zip(_v, [0]*len(_v) if self.last_stats is None else self.last_stats[_k])])

        # add stats that have just been produced to tensorboard / sacred
        for _k, _v in aggregate_stats.items():
            add_stat_fn(_k, _v)

        # collect stats for logging horizon
        self.stats.append(aggregate_stats)
        # update last stats
        self.last_stats = current_stats
        pass

    def log(self, log_directly=False):
        assert not log_directly, "log_directly not supported."
        logging_str = " Win rate: {}".format(_seq_mean([ stat["win_rate"] for stat in self.stats ]))\
                    + " Timeouts: {}".format(_seq_mean([ stat["timeouts"] for stat in self.stats ]))\
                    + " Restarts: {}".format(_seq_mean([ stat["restarts"] for stat in self.stats ]))

        # flush stats
        self.stats = []
        return logging_str
