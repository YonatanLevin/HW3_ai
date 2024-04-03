"# HW3_ai" 
TODO: 

Part 1
given 300 sec initially to explore. not needed in base model.
given 5 sec every turn to explore and choose action.

initially build a MCT.
Every node is an action.
The children of a node are all unconstrained actions.
during simulation act like sample_agent
In expansion, expend all actions.

In act, run mcts from the state, 
and return the best child of the root


Part 2:
Structure:
every node has a state, num of visits and actions.
Every action has list of resulting states and number of visits for every state.
The value of an action for a state is the weighted average value of resulting states.
The value of a state is the maximal value among the actions.

Backpropagation:
only on forward propagation path.
Implement with a stack

