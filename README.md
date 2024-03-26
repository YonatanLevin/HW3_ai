"# HW3_ai" 
TODO: 

given 300 sec initially to explore. not needed in base model.
given 5 sec every turn to explore and choose action.

initially build a MCT.
Every node is an action.
The children of a node are all unconstrained actions.
during simulation act like sample_agent
In expansion, expend all actions.

In act, run mcts from the state, 
and return the best child of the root
