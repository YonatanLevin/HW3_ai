import time

IDS = ["213125164", "325407054"]
from simulator import Simulator
import random
import math
from typing import List, Tuple
import itertools
from sample_agent import Agent as SampleAgent

CONSTRUCTOR_TIMEOUT = 60
ACTION_TIMEOUT = 5


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


class UCTNode:
    """
    A class for a single node
    """

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []

    def add_child(self, child_state, move):
        child = UCTNode(child_state, self, move)
        self.children.append(child)
        return child

    def select_child(self, simulator) -> UCTNode:
        return max(self.children, key=lambda child: child.uct_value(simulator))

    def expand(self, simulator, player_number, my_ships):
        state = simulator.state
        actions = {}
        collected_treasures = []
        for ship in my_ships:
            actions[ship] = set()
            neighboring_tiles = simulator.neighbors(state["pirate_ships"][ship]["location"])
            for tile in neighboring_tiles:
                actions[ship].add(("sail", ship, tile))
            if state["pirate_ships"][ship]["capacity"] > 0:
                for treasure in state["treasures"].keys():
                    if state["pirate_ships"][ship]["location"] in simulator.neighbors(
                            state["treasures"][treasure]["location"]) and treasure not in collected_treasures:
                        actions[ship].add(("collect", ship, treasure))
                        collected_treasures.append(treasure)
            for treasure in state["treasures"].keys():
                if (state["pirate_ships"][ship]["location"] == state["base"]
                        and state["treasures"][treasure]["location"] == ship):
                    actions[ship].add(("deposit", ship, treasure))
            for enemy_ship_name in state["pirate_ships"].keys():
                if (state["pirate_ships"][ship]["location"] == state["pirate_ships"][enemy_ship_name]["location"] and
                        player_number != state["pirate_ships"][enemy_ship_name]["player"]):
                    actions[ship].add(("plunder", ship, enemy_ship_name))
            actions[ship].add(("wait", ship))
        self.children = list(itertools.product(*actions.values()))



    def update(self, result):
        self.visits += 1
        self.wins += result

    def uct_value(self, simulator) -> float:
        if simulator.check_if_action_legal()
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)


class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """

    def __init__(self):
        raise NotImplementedError



class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.start = time.time()
        self.prev_actions = dict()
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.map = initial_state['map']
        self.base = initial_state['base']
        self.initial_state = initial_state
        self.treasure_rewards = {key: t['reward'] for key, t in initial_state['treasures'].items()}
        self.rows = len(self.map)
        self.columns = len(self.map[0])
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)

    def selection(self, UCT_tree: UCTNode, simulator, sample_agent: SampleAgent):
        current_node = UCT_tree
        while len(current_node.children) != 0:
            current_node = current_node.select_child(simulator)
            simulator.apply_action(current_node, 'player 1')
            simulator.apply_action(sample_agent.act(simulator.state), 'player 2')
        return current_node

    def expansion(self, parent_node: UCTNode):


    def simulation(self, node, player) -> int:
        """
        Preforms a random simulation
        """
        moves = node.state.get_moves()
        child_state = node.state.make_move(random.choice(moves))
        return self.simulation(UCTNode(child_state, node), -player)

    def backpropagation(self, node, simulation_result):
        while node is not None:
            node.update(simulation_result)
            node = node.parent

    def act(self, state):
        return self.mcts(state).move

    def get_actions(self, state):


    def mcts(self, state) -> UCTNode:
        root = UCTNode(state)
        simulator = Simulator(state)
        sample_agent = SampleAgent(state, 2)
        try:
            while True:
                if time.time() - self.start > CONSTRUCTOR_TIMEOUT - 55:
                    raise TimeoutError("")
                node = self.selection(root, simulator, sample_agent)
                self.expansion(node)
                result = self.simulation(node, node.state.player)
                self.backpropagation(node, result)
        except TimeoutError:
            pass

        return max(root.children, key=lambda child: child.wins / child.visits)  # No exploration
