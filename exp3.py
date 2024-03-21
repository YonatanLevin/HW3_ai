IDS = ["Your IDS here"]
from simulator import Simulator
import random


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
    A class for a single node. not mandatory to use but may help you.
    """
    def __init__(self):
        raise NotImplementedError


class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """
    def __init__(self):
        raise NotImplementedError


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)

    def selection(self, UCT_tree):
        raise NotImplementedError

    def expansion(self, UCT_tree, parent_node):
        raise NotImplementedError

    def simulation(self):
        raise NotImplementedError

    def backpropagation(self, simulation_result):
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError
