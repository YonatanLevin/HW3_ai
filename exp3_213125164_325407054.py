import time
from simulator import Simulator
import random
import math
from typing import List, Tuple
import itertools

IDS = ["213125164", "325407054"]

CONSTRUCTOR_TIMEOUT = 55
ACTION_TIMEOUT = 4.5
PLAYER_1 = 1
PLAYER_2 = 2
PLAYER_1_NAME = "player 1"
PLAYER_2_NAME = "player 2"


def check_time(start, timeout):
    if time.time() - start > timeout:
        raise TimeoutError("")


def heuristic(state, player_number, heuristic_name):
    return heuristic_name(state, player_number)


def heuristic_1(state, player_number):
    base = state["base"]
    total = 0
    for ship_name, ship_value in state["pirate_ships"].items():
        if ship_value["player"] != player_number:
            break
        treasures = [treasure for treasure in state["treasures"].values() if treasure["location"] == ship_name]
        reward = sum([treasure["reward"] for treasure in treasures])
        distance = abs(base[0] - ship_value["location"][0]) + abs(base[1] - ship_value["location"][1])
        total += reward / (distance + 1)
    return total


def hash_state(state):
    pirate_ships = sum(f'{k}{v["location"]}{v["capacity"]}{v["player"]};' for k, v in state["pirate_ships"].items())
    treasures = sum(f'{k}{v["location"]}{v["reward"]};' for k, v in state["treasures"].items())
    marine_ships = sum(f'{k}{v["index"]}{v["path"]};' for k, v in state["marine_ships"].items())
    return hash(pirate_ships + ';' + treasures + ';' + marine_ships + ';' + state["turns to go"])


def action_heuristic(move):
    if not move:
        return 0
    value = 0
    for atomic_action in move:
        if atomic_action[0] == 'collect':
            value += 3
        elif atomic_action[0] == 'deposit':
            value += 5
        elif atomic_action[0] == 'plunder':
            value += 2
        elif atomic_action[0] == 'wait':
            value -= 0.5
    return value


class Node:
    """
    A class for a single node
    """

    def __init__(self, state, player_number, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []
        self.player_number = player_number
        self.h = action_heuristic(move)


    def add_child(self, child_state, move):
        child = Node(child_state, self.player_number, self, move)
        self.children.append(child)
        return child

    def select_child(self, simulator, moves):
        return max(self.children, key=lambda child: child.uct_value(simulator, moves))

    def expand(self, actions):
        for action in actions:
            self.add_child(self.state, action)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def uct_value(self, simulator, moves) -> float:
        if not check_if_action_legal(simulator, self.move, PLAYER_1 if self.player_number == PLAYER_1 else PLAYER_2,
                                     moves):
            return float('-inf')
        if self.visits == 0:
            return 9999 + self.h
        return (self.wins + self.h) / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)


class Agent:
    def __init__(self, initial_state, player_number):
        self.start = time.time()
        self.prev_actions = dict()
        self.ids = IDS
        self.player_number = player_number
        self.initial_state = initial_state
        self.my_ships = get_my_ships(initial_state, player_number)
        self.his_ships = get_my_ships(initial_state, PLAYER_1 if player_number == PLAYER_2 else PLAYER_2)
        # self.actions_by_location = get_actions_by_location(initial_state, player_number)
        self.moves_by_location = get_neighbor_dict(initial_state['map'])
        self.my_sail_actions = get_sail_actions(initial_state, player_number, self.moves_by_location)
        self.his_sail_actions = get_sail_actions(initial_state, PLAYER_1 if player_number == PLAYER_2 else PLAYER_2
                                                 , self.moves_by_location)

    def selection(self, node: Node, simulator: Simulator, sample_agent, start_time):
        """
        Select the best child nodes
        :param node: node to start from
        :param simulator: instance of the simulator
        :param sample_agent: opponent agent
        :param start_time: the start time of the time limited phase
        :return: the best child nodes
        """

        # initialize the current node
        current_node = node

        turns = 0

        # while the current node has children
        while len(current_node.children) != 0:
            check_time(start_time, ACTION_TIMEOUT)
            turns += 1
            # select the best child node

            # apply the action of the current node
            if self.player_number == PLAYER_1:
                current_node = current_node.select_child(simulator, self.moves_by_location)
                simulator.apply_action(current_node.move, self.player_number)
                simulator.add_treasure()
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_2)
            else:
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_1)
                simulator.add_treasure()
                current_node = current_node.select_child(simulator, self.moves_by_location)
                simulator.apply_action(current_node.move, self.player_number)
            simulator.add_treasure()
            simulator.check_collision_with_marines()
            simulator.move_marines()

        # return the current node
        return current_node, turns

    def expansion(self, parent_node: Node, simulator: Simulator):
        """
        Expand the parent node
        :param simulator:
        :param parent_node: parent node
        """

        # getting all the possible actions
        action_list = self.get_actions(simulator)

        # expanding the parent node
        parent_node.expand(action_list)

    def simulation(self, node, simulator: Simulator, sample_agent, turns, turns_to_go, start) -> int:
        """
        Preforms a random simulation
        """

        # initialize the current node
        current_node = node

        # running the simulation
        my_sample_agent = MySampleAgent(current_node.state, self.player_number, self.moves_by_location, self.my_ships
                                        , self.my_sail_actions)
        for i in range(turns_to_go - turns):
            check_time(start, ACTION_TIMEOUT)

            if self.player_number == PLAYER_1:
                simulator.apply_action(my_sample_agent.act(simulator.state), self.player_number)
                simulator.add_treasure()
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_2)
            else:
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_1)
                simulator.add_treasure()
                simulator.apply_action(my_sample_agent.act(simulator.state), self.player_number)
            simulator.add_treasure()
            simulator.check_collision_with_marines()
            simulator.move_marines()

        score = simulator.get_score()
        return (score[PLAYER_1_NAME if self.player_number == PLAYER_1 else PLAYER_2_NAME] -
                score[PLAYER_2_NAME if self.player_number == PLAYER_1 else PLAYER_1_NAME])

    def backpropagation(self, node, simulation_result):
        while node is not None:
            node.update(simulation_result)
            node = node.parent

    def act(self, state):
        return self.mcts(state).move

    def get_actions(self, simulator):
        actions = {}
        collected_treasures = []
        state = simulator.state
        for ship in self.my_ships:
            actions[ship] = get_actions_for_ship(ship, state, collected_treasures, simulator, self.player_number,
                                                 self.moves_by_location)
        all_combinations = list(itertools.product(*actions.values()))
        return all_combinations

    def mcts(self, state) -> Node:
        start = time.time()
        root = Node(state, self.player_number)
        turns_to_go = state["turns to go"] // 2
        count_simulations = 0
        try:
            while True:
                check_time(start, ACTION_TIMEOUT)
                simulator = Simulator(state)
                sample_agent = MySampleAgent(state, PLAYER_1 if self.player_number == PLAYER_2 else PLAYER_2,
                                             self.moves_by_location, self.his_ships, self.his_sail_actions,
                                             )
                check_time(start, ACTION_TIMEOUT)
                node, turns = self.selection(root, simulator, sample_agent, start)
                if turns >= turns_to_go:
                    break
                self.expansion(node, simulator)
                result = self.simulation(node, simulator, sample_agent, turns, turns_to_go, start)
                self.backpropagation(node, result)
                count_simulations += 1
        except TimeoutError:
            pass
        print(f'count_simulations: {count_simulations}')

        if len(root.children) == 0:
            return root

        return max(root.children,
                   key=lambda child: child.wins / child.visits if child.visits > 0 else 0)


# -------------------------------------------- UCT Node ------------------------------------------------------------


class UCTNode:
    """
    A class for a single node
    """

    def __init__(self, state, player_number, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []
        self.player_number = player_number

    def add_child(self, child_state, move):
        child = UCTNode(child_state, self.player_number, self, move)
        self.children.append(child)
        return child

    def select_child(self, simulator, moves):
        return max(self.children, key=lambda child: child.uct_value(simulator, moves))

    def expand(self, actions):
        for action in actions:
            self.add_child(self.state, action)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def uct_value(self, simulator, moves) -> float:
        if not check_if_action_legal(simulator, self.move, PLAYER_1 if self.player_number == PLAYER_1 else PLAYER_2,
                                     moves):
            return float('-inf')
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)


class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """

    def __init__(self):
        raise NotImplementedError


# -------------------------------------------- UCT Agent --------------------------------------------


def get_my_ships(state, player_number) -> List[str]:
    """
    Get all the ships of the player
    :param state: current state
    :param player_number: player number
    :return:
    """

    my_ships = []
    for ship_name, ship in state['pirate_ships'].items():
        if ship['player'] == player_number:
            my_ships.append(ship_name)
    return my_ships


def get_actions_for_ship(ship, state, collected_treasures, simulator, player_number, neighbors_dict):
    actions = set()
    neighboring_tiles = neighbors_dict[state["pirate_ships"][ship]["location"]]
    actions.update(('sail', ship, tile) for tile in neighboring_tiles)
    actions.update(get_collect_actions(ship, state, collected_treasures, simulator))
    actions.update(get_deposit_actions(ship, state))
    actions.update(get_plunder_actions(ship, state, player_number))
    actions.add(("wait", ship))
    return actions


def get_collect_actions(ship, state, collected_treasures, simulator, neighbors_dict):
    actions = set()
    if state["pirate_ships"][ship]["capacity"] > 0:
        for treasure in state["treasures"].keys():
            treasure_loc = state["treasures"][treasure]["location"]
            if type(treasure_loc) == str:
                continue
            if pirate_loc in neighbors_dict[
                treasure_loc] and treasure not in collected_treasures:
                actions.add(("collect", ship, treasure))
    return actions


def get_deposit_actions(ship, state):
    actions = set()
    for treasure in state["treasures"].keys():
        if (state["pirate_ships"][ship]["location"] == state["base"]
                and state["treasures"][treasure]["location"] == ship):
            actions.add(("deposit", ship, treasure))
    return actions


def get_plunder_actions(ship, state, name):
    actions = set()
    for enemy_ship_name in state["pirate_ships"].keys():
        if (state["pirate_ships"][ship]["location"] == state["pirate_ships"][enemy_ship_name]["location"] and
                name != state["pirate_ships"][enemy_ship_name]["player"]):
            actions.add(("plunder", ship, enemy_ship_name))
    return actions


def get_actions_by_location(state, player_number, neighbors_dict):
    actions_by_location = dict()
    for ship_name, ship in state["pirate_ships"].items():
        if ship["player"] == player_number:
            actions_by_location[ship["location"]] = get_actions_for_ship(ship_name, state, [], Simulator(state),
                                                                         player_number, neighbors_dict)
    return actions_by_location


def get_neighbor_dict(map):
    neighbors = dict()
    for i in range(len(map)):
        for j in range(len(map[0])):
            neighbors_in = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
            for neighbor in tuple(neighbors_in):
                if neighbor[0] < 0 or neighbor[0] >= len(map) or neighbor[1] < 0 or neighbor[1] >= \
                        len(map[0]) or map[neighbor[0]][neighbor[1]] == 'I':
                    neighbors_in.remove(neighbor)
            neighbors[(i, j)] = neighbors_in
    return neighbors


def get_sail_actions(state, player, neighbors):
    sail_actions = {}
    map = state["map"]
    for ship_name, ship in state["pirate_ships"].items():
        if ship["player"] == player:
            for i in range(len(map)):
                for j in range(len(map[0])):
                    if map[i][j] != 'I':
                        sail_actions[(ship_name, (i, j))] = [('sail', ship_name, neighbor)
                                                             for neighbor in neighbors[(i, j)]]
    return sail_actions


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.start = time.time()
        self.prev_actions = dict()
        self.ids = IDS
        self.player_number = player_number
        self.initial_state = initial_state
        self.my_ships = get_my_ships(initial_state, player_number)
        self.his_ships = get_my_ships(initial_state, PLAYER_1 if player_number == PLAYER_2 else PLAYER_2)
        # self.actions_by_location = get_actions_by_location(initial_state, player_number)
        self.moves_by_location = get_neighbor_dict(initial_state['map'])
        self.my_sail_actions = get_sail_actions(initial_state, player_number, self.moves_by_location)
        self.his_sail_actions = get_sail_actions(initial_state, PLAYER_1 if player_number == PLAYER_2 else PLAYER_2
                                                 , self.moves_by_location)

    def selection(self, node: UCTNode, simulator: Simulator, sample_agent, start_time):
        """
        Select the best child nodes
        :param node: node to start from
        :param simulator: instance of the simulator
        :param sample_agent: opponent agent
        :return: the best child nodes
        """

        # initialize the current node
        current_node = node

        turns = 0

        # while the current node has children
        while len(current_node.children) != 0:
            check_time(start_time, ACTION_TIMEOUT)
            turns += 1
            # select the best child node

            # apply the action of the current node
            if self.player_number == PLAYER_1:
                current_node = current_node.select_child(simulator, self.moves_by_location)
                simulator.apply_action(current_node.move, self.player_number)
                simulator.add_treasure()
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_2)
            else:
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_1)
                simulator.add_treasure()
                current_node = current_node.select_child(simulator, self.moves_by_location)
                simulator.apply_action(current_node.move, self.player_number)
            simulator.add_treasure()
            simulator.check_collision_with_marines()
            simulator.move_marines()

        # return the current node
        return current_node, turns

    def expansion(self, parent_node: UCTNode, simulator: Simulator):
        """
        Expand the parent node
        :param simulator:
        :param parent_node: parent node
        """

        # getting all the possible actions
        action_list = self.get_actions(simulator)

        # expanding the parent node
        parent_node.expand(action_list)

    def simulation(self, node, simulator: Simulator, sample_agent, turns, turns_to_go, start) -> int:
        """
        Preforms a random simulation
        """

        # initialize the current node
        current_node = node

        # running the simulation
        my_sample_agent = MySampleAgent(current_node.state, self.player_number, self.moves_by_location, self.my_ships
                                        , self.my_sail_actions)
        for i in range(turns_to_go - turns):
            check_time(start, ACTION_TIMEOUT)

            if self.player_number == PLAYER_1:
                simulator.apply_action(my_sample_agent.act(simulator.state), self.player_number)
                simulator.add_treasure()
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_2)
            else:
                simulator.apply_action(sample_agent.act(simulator.state), PLAYER_1)
                simulator.add_treasure()
                simulator.apply_action(my_sample_agent.act(simulator.state), self.player_number)
            simulator.add_treasure()
            simulator.check_collision_with_marines()
            simulator.move_marines()

        score = simulator.get_score()
        return (score[PLAYER_1_NAME if self.player_number == PLAYER_1 else PLAYER_2_NAME] -
                score[PLAYER_2_NAME if self.player_number == PLAYER_1 else PLAYER_1_NAME])

    def backpropagation(self, node, simulation_result):
        while node is not None:
            node.update(simulation_result)
            node = node.parent

    def act(self, state):
        return self.mcts(state).move

    def get_actions(self, simulator):
        actions = {}
        collected_treasures = []
        state = simulator.state
        for ship in self.my_ships:
            actions[ship] = get_actions_for_ship(ship, state, collected_treasures, simulator, self.player_number,
                                                 self.moves_by_location)
        all_combinations = list(itertools.product(*actions.values()))
        return all_combinations

    def mcts(self, state) -> UCTNode:
        start = time.time()
        root = UCTNode(state, self.player_number)
        turns_to_go = state["turns to go"]
        try:
            while True:
                check_time(start, ACTION_TIMEOUT)
                simulator = Simulator(state)
                sample_agent = MySampleAgent(state, PLAYER_1 if self.player_number == PLAYER_2 else PLAYER_2,
                                             self.moves_by_location, self.his_ships, self.his_sail_actions,
                                             )
                check_time(start, ACTION_TIMEOUT)
                node, turns = self.selection(root, simulator, sample_agent, start)
                if turns >= turns_to_go:
                    break
                self.expansion(node, simulator)
                result = self.simulation(node, simulator, sample_agent, turns, turns_to_go, start)
                self.backpropagation(node, result)
        except TimeoutError:
            pass

        if len(root.children) == 0:
            return root

        return max(root.children,
                   key=lambda child: child.wins / child.visits if child.visits > 0 else 0)


def check_if_action_legal(simulator, action, player, moves_by_location):
    def _is_move_action_legal(move_action, player):
        pirate_name = move_action[1]
        if pirate_name not in simulator.state['pirate_ships'].keys():
            return False
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        l1 = simulator.state['pirate_ships'][pirate_name]['location']
        l2 = move_action[2]
        if l2 not in moves_by_location[l1]:
            return False
        return True

    def _is_collect_action_legal(collect_action, player):
        pirate_name = collect_action[1]
        treasure_name = collect_action[2]
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        # check adjacent position
        l1 = simulator.state['treasures'][treasure_name]['location']
        if type(l1) == str or simulator.state['pirate_ships'][pirate_name]['location'] not in moves_by_location[l1]:
            return False
        # check ship capacity
        if simulator.state['pirate_ships'][pirate_name]['capacity'] <= 0:
            return False
        return True

    def _is_deposit_action_legal(deposit_action, player):
        pirate_name = deposit_action[1]
        treasure_name = deposit_action[2]
        # check same position
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        if simulator.state["pirate_ships"][pirate_name]["location"] != simulator.base_location:
            return False
        if simulator.state['treasures'][treasure_name]['location'] != pirate_name:
            return False
        return True

    def _is_plunder_action_legal(plunder_action, player):
        pirate_1_name = plunder_action[1]
        pirate_2_name = plunder_action[2]
        if player != simulator.state["pirate_ships"][pirate_1_name]["player"]:
            return False
        if simulator.state["pirate_ships"][pirate_1_name]["location"] != simulator.state["pirate_ships"][pirate_2_name][
            "location"]:
            return False
        return True

    def _is_action_mutex(global_action):
        assert type(
            global_action) == tuple, "global action must be a tuple"
        # one action per ship
        if len(set([a[1] for a in global_action])) != len(global_action):
            return True
        # collect the same treasure
        collect_actions = [a for a in global_action if a[0] == 'collect']
        if len(collect_actions) > 1:
            treasures_to_collect = set([a[2] for a in collect_actions])
            if len(treasures_to_collect) != len(collect_actions):
                return True

        return False

    # def get_possible_collect_action(simulator, player):
    #     for pirate in simulator.state['pirate_ships'].keys():
    #         if simulator.state['pirate_ships'][pirate]['player'] == player:
    #             for treasure in simulator.state['treasures'].keys():
    #                 if simulator.state['pirate_ships'][pirate]['location'] in simulator.neighbors(
    #                         simulator.state['treasures'][treasure]['location']):
    #                     return 'collect', pirate, treasure
    #     return None

    players_pirates = [pirate for pirate in simulator.state['pirate_ships'].keys() if
                       simulator.state['pirate_ships'][pirate]['player'] == player]

    if len(action) != len(players_pirates):
        return False
    for atomic_action in action:
        # if atomic_action == 'collect':
        #     atomic_action = get_possible_collect_action(simulator, player)
        #     if atomic_action is None:
        #         return False
        # trying to act with a pirate that is not yours
        if atomic_action[1] not in players_pirates:
            return False
        # illegal sail action
        # if atomic_action[0] == 'sail':
        #     if not _is_move_action_legal(atomic_action, player):
        #         return False
        # illegal collect action
        elif atomic_action[0] == 'collect':
            if not _is_collect_action_legal(atomic_action, player):
                return False
        # illegal deposit action
        elif atomic_action[0] == 'deposit':
            if not _is_deposit_action_legal(atomic_action, player):
                return False
        # illegal plunder action
        elif atomic_action[0] == "plunder":
            if not _is_plunder_action_legal(atomic_action, player):
                return False
        # elif atomic_action[0] != 'wait':
        #     return False
    # check mutex action
    if _is_action_mutex(action):
        return False
    return True


##########################################################################################################
class MySampleAgent:
    def __init__(self, initial_state, player_number, neighbors_dict, my_ships, sail_actions):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.neighbors_dict = neighbors_dict
        self.my_ships = my_ships
        self.sail_actions = sail_actions
        self.simulator = Simulator(initial_state)
        # for ship_name, ship in initial_state['pirate_ships'].items():
        #     if ship['player'] == player_number:
        #         self.my_ships.append(ship_name)

    def act(self, state):
        actions = {}
        self.simulator.set_state(state)
        collected_treasures = []
        for ship in self.my_ships:
            actions[ship] = set()
            ship_loc = state["pirate_ships"][ship]["location"]
            actions[ship].update(self.sail_actions[(ship, ship_loc)])
            if state["pirate_ships"][ship]["capacity"] > 0:
                for treasure in state["treasures"].keys():
                    treasure_loc = state["treasures"][treasure]["location"]
                    if type(treasure_loc) == str:
                        continue
                    if (ship_loc in self.neighbors_dict[treasure_loc] and
                            treasure not in collected_treasures):
                        actions[ship].add(("collect", ship, treasure))
                        collected_treasures.append(treasure)
            for treasure in state["treasures"].keys():
                if (ship_loc == state["base"]
                        and state["treasures"][treasure]["location"] == ship):
                    actions[ship].add(("deposit", ship, treasure))
            for enemy_ship_name in state["pirate_ships"].keys():
                if (ship_loc == state["pirate_ships"][enemy_ship_name]["location"] and
                        self.player_number != state["pirate_ships"][enemy_ship_name]["player"]):
                    actions[ship].add(("plunder", ship, enemy_ship_name))
            actions[ship].add(("wait", ship))

        whole_action = []
        for atomic_actions in actions.values():
            for action in atomic_actions:
                if action[0] == "deposit":
                    whole_action.append(action)
                    break
                if action[0] == "collect":
                    whole_action.append(action)
                    break
            else:
                whole_action.append(random.choice(list(atomic_actions)))
        return whole_action
