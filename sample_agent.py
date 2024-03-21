import random
from simulator import Simulator

IDS = ['AI2']


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
        actions = {}
        self.simulator.set_state(state)
        collected_treasures = []
        for ship in self.my_ships:
            actions[ship] = set()
            neighboring_tiles = self.simulator.neighbors(state["pirate_ships"][ship]["location"])
            for tile in neighboring_tiles:
                actions[ship].add(("sail", ship, tile))
            if state["pirate_ships"][ship]["capacity"] > 0:
                for treasure in state["treasures"].keys():
                    if state["pirate_ships"][ship]["location"] in self.simulator.neighbors(
                            state["treasures"][treasure]["location"]) and treasure not in collected_treasures:
                        actions[ship].add(("collect", ship, treasure))
                        collected_treasures.append(treasure)
            for treasure in state["treasures"].keys():
                if (state["pirate_ships"][ship]["location"] == state["base"]
                        and state["treasures"][treasure]["location"] == ship):
                    actions[ship].add(("deposit", ship, treasure))
            for enemy_ship_name in state["pirate_ships"].keys():
                if (state["pirate_ships"][ship]["location"] == state["pirate_ships"][enemy_ship_name]["location"] and
                        self.player_number != state["pirate_ships"][enemy_ship_name]["player"]):
                    actions[ship].add(("plunder", ship, enemy_ship_name))
            actions[ship].add(("wait", ship))

        while True:
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
            whole_action = tuple(whole_action)
            if self.simulator.check_if_action_legal(whole_action, self.player_number):
                return whole_action
