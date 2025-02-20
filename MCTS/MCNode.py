
from Game.Game import *
from Conventions import *
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
    
    
    def __init__(self, state: AbstractState, parent: MCNode):
        self.state = state
        self.parent = parent
    
    
    # Generate the children of this node.
    def expand(self, game: AbstractGame, dynamics_network: NeuralNetwork):
        
        assert len(self.children) == 0, 'Unexpectedly regenerated children of node.'
        
        action_space = game.action_space()
        
        for action in action_space:
            network_input = dynamics_network_input(self.state, action)
            next_abstract_state = dynamics_network(network_input)
            child = MCNode(next_abstract_state, self)
            self.children[action] = child
    
    
    # Randomly choose a child and return it. Uniform distribution.
    # Save the action for later. It's used during backpropagation.
    def uniform_get_random_child(self) -> MCNode:
        self.action_taken = uniform_choice(self.children.keys())
        child = self.children[self.action_taken]
        return child
    
    
    # Randomly select a child and return it.
    # Probabilities are proportional to the child's visit
    # count, so more explored children are favored. Hence,
    # is suitable to select actual action after MCTS.
    def biased_get_random_action(self) -> Action:
        actions = list(self.children.keys())  # TODO: This is probably quite slow. Find better way to accomplish the same thing.
        weights = [ self.children[action].visits_to_self for action in actions ]
        return weighted_choice(actions, weights=weights)
    
    
    # A node is considered a leaf if it has no children.
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    
    # u(a) is the exploration bonus of action a
    def u(self, action: Action) -> float:
        N_sa = self.visit_counts[action]
        return self._c * math.sqrt( math.log2(self.visits_to_self) / (1 + N_sa) )
    
    
    # Q(a) is the value of doing action a
    def Q(self, action: Action) -> float:
        return self.visit_counts[action] / self.sum_evaluation
    
    
    # Backpropagate the value up through the tree.
    # Discount by multiplying by the discount factor (e.g. 0.95) at each step.
    def backpropagate(self, value: float, discount_factor: float):
        
        self.sum_evaluation += value
        self.visits_to_self += 1
        
        if self.visit_counts[self.action_taken] is None:
            self.visit_counts[self.action_taken] = 0
        self.visit_counts[self.action_taken] += 1
        
        if self.parent == None:
            return
        
        self.parent.backpropagate(value * discount_factor, discount_factor)
