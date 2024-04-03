from simulator import Simulator
import ex3_213125164_325407054
import sample_agent
from copy import deepcopy
import time
import matplotlib.pyplot as plt

CONSTRUCTOR_TIMEOUT = 60
ACTION_TIMEOUT = 5
DIMENSIONS = (7, 7)
PENALTY = 10000


class Game:
    """
    This class plays the game for you. You are given a sample agent to play against.
    """
    def __init__(self, an_input):
        self.initial_state = deepcopy(an_input)
        self.simulator = Simulator(self.initial_state)
        self.ids = []
        self.agents = []
        self.score = [0, 0]

    def initiate_agent(self, module, player_number, UCT_flag=False):
        """
        :param UCT_flag: Uses UCT_Agent instead of general one
        :return: agent
        """
        start = time.time()
        if UCT_flag:
            agent = module.UCTAgent(self.initial_state, player_number)
        else:
            agent = module.Agent(self.initial_state, player_number)
        if time.time() - start > CONSTRUCTOR_TIMEOUT:
            raise ValueError(f'agent timed out on constructor!')
        return agent

    def get_action(self, agent, player):
        start = time.time()
        action = agent.act(self.simulator.get_state())
        finish = time.time()
        if finish - start > ACTION_TIMEOUT:
            self.score[player] -= PENALTY
            raise ValueError(f'{self.ids[player]} timed out on action!')
        return action

    def play_episode(self, swapped=False):
        length_of_episode = int(self.initial_state["turns to go"]/2)
        for i in range(length_of_episode):
            print(f'Turn {i + 1}')
            for number, agent in enumerate(self.agents):
                try:
                    action = self.get_action(agent, number)
                except (AssertionError, ValueError) as e:
                    print(e)
                    self.score[number] -= PENALTY
                    return
                try:
                    self.simulator.act(action, number + 1)
                except (AssertionError, ValueError):
                    print(f'{agent.ids} chose illegal action!')
                    self.score[number] -= PENALTY
                    return
                print(f"{agent.ids} chose {action}")
            self.simulator.check_collision_with_marines()
            self.simulator.move_marines()
            print(self.simulator.get_score())
            print(f"-----")
        if not swapped:
            self.score[0] += self.simulator.get_score()['player 1']
            self.score[1] += self.simulator.get_score()['player 2']
        else:
            self.score[0] += self.simulator.get_score()['player 2']
            self.score[1] += self.simulator.get_score()['player 1']

            print(f'***********  end of round!  ************ \n \n')

    def play_game(self):
        """
        When initiating the agents in this function, you can set UCT_flag to True in initiate_agent(), when not using
        the general agent. You may also use an agent of your own, instead of sample agent.
        """
        print(f'***********  starting a first round!  ************ \n \n')
        self.agents = [self.initiate_agent(ex3_213125164_325407054, 1, UCT_flag=True),
                       self.initiate_agent(sample_agent, 2)]
        self.ids = ['Your agent', 'Rival agent']
        self.play_episode()
        print(self.simulator.state)

        print(f'***********  starting a second round!  ************ \n \n')
        self.simulator = Simulator(self.initial_state)

        self.agents = [self.initiate_agent(sample_agent, 1),
                       self.initiate_agent(ex3_213125164_325407054, 2, UCT_flag=True)]
        self.ids = ['Rival agent', 'Your agent']
        self.play_episode(swapped=True)
        print(f'end of game!')
        return self.score


def main():
    an_input = {
        "map": [
            ['S', 'S', 'I', 'S', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S', 'S', 'S', 'S'],
            ['B', 'S', 'S', 'S', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S', 'S', 'I', 'S'],
            ['S', 'S', 'I', 'S', 'S', 'I', 'S'],
            ['S', 'S', 'S', 'S', 'S', 'I', 'S'],
            ['S', 'S', 'S', 'S', 'S', 'I', 'I']
        ],
        "base": (2, 0),
        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2,
                                           "player": 1},
                         'pirate_ship_2': {"location": (2, 0,),
                                           "capacity": 2,
                                           "player": 1},
                         'pirate_ship_3': {"location": (2, 0,),
                                           "capacity": 2,
                                           "player": 2},
                         'pirate_ship_4': {"location": (2, 0,),
                                           "capacity": 2,
                                           "player": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "reward": 4}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (2, 4)]},
                         'marine_2': {"index": 0,
                                      "path": [(2, 5), (2, 4), (3, 4), (4, 4)]}
                         },
        "turns to go": 200
    }
    start = time.time()

    # to calculate the expected value
    final_score = [0, 0]
    scores = [[], []]
    difference = []

    num_runs = 3
    for i in range(num_runs):
        print(f'***********  Starting run number {i}  ************ \n \n')
        game = Game(an_input)
        results = game.play_game()
        print(f'Intermediate score: {results}')
        final_score[0] += results[0]
        final_score[1] += results[1]
        scores[0].append(results[0])
        scores[1].append(results[1])
        difference.append(results[0] - results[1])

    print(f'Score for {ex3_213125164_325407054.IDS} is {final_score[0]/num_runs}, score for {sample_agent.IDS} is {final_score[1]/num_runs}')
    print(f'Average Difference: {sum(difference)/num_runs}')
    print(f'Time taken: {time.time() - start}')

    # plotting the results
    plt.plot(scores[0], color='g', linestyle='-', label='UCT agent')
    plt.plot(scores[1], color='r', linestyle='-', label='Sample agent')
    # plotting the differences between value of the agents
    plt.plot(difference, color='b', linestyle='-', label='Difference')
    # plotting the line representing the average value of the agents
    plt.axhline(y=final_score[0]/num_runs, color='g', linestyle='--', label='Average UCT agent score')
    plt.axhline(y=final_score[1]/num_runs, color='r', linestyle='--', label='Average sample agent score')
    # plotting the line representing the average difference between the agents
    plt.axhline(y=sum(difference)/num_runs, color='b', linestyle='--', label='Average difference')

    plt.legend()
    plt.xlabel('Number of run')
    plt.ylabel('Score')
    plt.title('Scores of the agents')
    plt.savefig('scores_10runs.png')


if __name__ == '__main__':
    main()
