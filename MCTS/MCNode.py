
from Game.Game import *
from MCTS.Conventions import *
from MCTS.MCTSTypes import *
from NeuralNetwork.NeuralNetwork import *

import math
from random import choice as uniform_choice, choices as weighted_choice

type MCNode = MCNode

class MCNode():
    
    # Constant in u(s, a) evaluation
    _c = 1
    
    state: AbstractState
    children: dict[Action, MCNode] = {}
    
    parent: MCNode
    action_taken: Action
    
    visits_to_self = 0
    visit_counts: dict[Action, int] = {}
    sum_evaluation: float = 0

    
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_counts = {}
        self.sum_evaluation = {}   
        self.action_taken = None 
        self.visits_to_self = 0  
    
    
    def expand(self, game, dynamics_network):
        if self.children:
            return  
        for action in game.action_space():
            network_input = dynamics_network_input(self.state, action)
            next_state = dynamics_network.predict(network_input)
            
            new_node = MCNode(next_state, parent=self)
            
            self.children[action] = new_node
            self.visit_counts[action] = 0
            self.sum_evaluation[action] = 0.0  

    
    
    # Randomly choose a child and return it. Uniform distribution.
    # Save the action for later. It's used during backpropagation.
    def uniform_get_random_child(self) -> MCNode:
        self.action_taken = uniform_choice(list(self.children.keys()))

        child = self.children[self.action_taken]
        return child
    
    
    # Randomly select a child and return it.
    # Probabilities are proportional to the child's visit
    # count, so more explored children are favored. Hence,
    # is suitable to select actual action after MCTS.
    def biased_get_random_action(self) -> Action:
        actions = list(self.children.keys())
        weights = [self.children[action].visits_to_self for action in actions]
        if sum(weights) == 0:
            return uniform_choice(actions)
        return weighted_choice(actions, weights=weights)

    
    
    # A node is considered a leaf if it has no children.
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    
    # u(a) is the exploration bonus of action a
    def u(self, action: Action) -> float:
        N_sa = self.visit_counts.get(action, 0)
        total_visits = max(self.visits_to_self, 1)
        return self._c * math.sqrt(math.log2(total_visits) / (1 + N_sa))

    
    
    # Q(a) is the value of doing action a
    def Q(self, action: Action) -> float:
        if self.visit_counts.get(action, 0) == 0:
            return 0.0
        return self.sum_evaluation[action] / self.visit_counts[action]


    
    
    # Backpropagate the value up through the tree.
    # Discount by multiplying by the discount factor (e.g. 0.95) at each step.
    def backpropagate(self, evaluation, discount_factor):
        if self.action_taken is not None:
            if self.visit_counts.get(self.action_taken) is None:
                self.visit_counts[self.action_taken] = 0
        if self.parent is not None:
            self.parent.backpropagate(evaluation, discount_factor)
