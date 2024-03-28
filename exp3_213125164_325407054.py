import time
from simulator import Simulator
import random
import math
from typing import List, Tuple
import itertools
from sample_agent import Agent as SampleAgent

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


class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)

    def act(self, state):
        raise NotImplementedError


# -------------------------------------------- UCT Node --------------------------------------------


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

    def select_child(self, simulator):
        return max(self.children, key=lambda child: child.uct_value(simulator))

    def expand(self, actions):
        for action in actions:
            self.add_child(self.state, action)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def uct_value(self, simulator) -> float:
        if not check_if_action_legal(simulator, self.move, PLAYER_1 if self.player_number == PLAYER_2 else PLAYER_2):
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


def get_actions_for_ship(ship, state, collected_treasures, simulator, player_number):
    actions = set()
    neighboring_tiles = simulator.neighbors(state["pirate_ships"][ship]["location"])
    actions.update(('sail', ship, tile) for tile in neighboring_tiles)
    actions.update(get_collect_actions(ship, state, collected_treasures, simulator))
    actions.update(get_deposit_actions(ship, state))
    actions.update(get_plunder_actions(ship, state, player_number))
    actions.add(("wait", ship))
    return actions


def get_collect_actions(ship, state, collected_treasures, simulator):
    actions = set()
    if state["pirate_ships"][ship]["capacity"] > 0:
        for treasure in state["treasures"].keys():
            if state["pirate_ships"][ship]["location"] in simulator.neighbors(
                    state["treasures"][treasure]["location"]) and treasure not in collected_treasures:
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


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.start = time.time()
        self.prev_actions = dict()
        self.ids = IDS
        self.player_number = player_number
        self.initial_state = initial_state
        self.my_ships = get_my_ships(initial_state, player_number)

    def selection(self, node: UCTNode, simulator: Simulator, sample_agent: SampleAgent):
        """
        Select the best child nodes
        :param node: node to start from
        :param simulator: instance of the simulator
        :param sample_agent: opponent agent
        :return: the best child nodes
        """

        # initialize the current node
        current_node = node

        # while the current node has children
        while len(current_node.children) != 0:
            check_time(self.start, ACTION_TIMEOUT)

            # select the best child node
            current_node = current_node.select_child(simulator)

            # apply the action of the current node
            simulator.apply_action(current_node.move, self.player_number)
            simulator.apply_action(sample_agent.act(simulator.state),
                                   PLAYER_1 if self.player_number == PLAYER_2 else PLAYER_2)

        # return the current node
        return current_node

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

    def simulation(self, node, simulator: Simulator, sample_agent: SampleAgent) -> int:
        """
        Preforms a random simulation
        """

        # initialize the current node
        current_node = node

        # running the simulation
        my_sample_agent = SampleAgent(current_node.state, self.player_number)
        for i in range(5):
            check_time(self.start, ACTION_TIMEOUT)

            # apply the action of the current node
            simulator.apply_action(my_sample_agent.act(simulator.state), self.player_number)
            simulator.apply_action(sample_agent.act(simulator.state),
                                   PLAYER_1 if self.player_number == PLAYER_2 else PLAYER_2)

        score = simulator.get_score()
        return (score[PLAYER_1_NAME if self.player_number == PLAYER_1 else PLAYER_2_NAME] -
                score[PLAYER_2_NAME if self.player_number == PLAYER_1 else PLAYER_2_NAME])

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
            actions[ship] = get_actions_for_ship(ship, state, collected_treasures, simulator, self.player_number)
        all_combinations = list(itertools.product(*actions.values()))
        return all_combinations

    def mcts(self, state) -> UCTNode:
        root = UCTNode(state, self.player_number)
        try:
            while True:
                check_time(self.start, ACTION_TIMEOUT)
                simulator = Simulator(state)
                sample_agent = SampleAgent(state, PLAYER_1 if self.player_number == PLAYER_2 else PLAYER_2)
                check_time(self.start, ACTION_TIMEOUT)
                self.node = self.selection(root, simulator, sample_agent)
                self.expansion(self.node, simulator)
                result = self.simulation(self.node, simulator, sample_agent)
                self.backpropagation(self.node, result)
        except TimeoutError:
            pass

        if len(root.children) == 0:
            return root

        return max(root.children,
                   key=lambda child: child.wins / child.visits if child.visits > 0 else 0)  # No exploration


def check_if_action_legal(simulator, action, player):
    def _is_move_action_legal(move_action, player):
        pirate_name = move_action[1]
        if pirate_name not in simulator.state['pirate_ships'].keys():
            return False
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        l1 = simulator.state['pirate_ships'][pirate_name]['location']
        l2 = move_action[2]
        if l2 not in simulator.neighbors(l1):
            return False
        return True

    def _is_collect_action_legal(collect_action, player):
        pirate_name = collect_action[1]
        treasure_name = collect_action[2]
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        # check adjacent position
        l1 = simulator.state['treasures'][treasure_name]['location']
        if simulator.state['pirate_ships'][pirate_name]['location'] not in simulator.neighbors(l1):
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

    players_pirates = [pirate for pirate in simulator.state['pirate_ships'].keys() if
                       simulator.state['pirate_ships'][pirate]['player'] == player]

    if len(action) != len(players_pirates):
        return False
    for atomic_action in action:
        # trying to act with a pirate that is not yours
        if atomic_action[1] not in players_pirates:
            return False
        # illegal sail action
        if atomic_action[0] == 'sail':
            if not _is_move_action_legal(atomic_action, player):
                return False
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
        elif atomic_action[0] != 'wait':
            return False
    # check mutex action
    if _is_action_mutex(action):
        return False
    return True
