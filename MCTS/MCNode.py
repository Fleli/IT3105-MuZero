
from Game.Game import *
from MCTS.Conventions import *
from MCTS.MCTSTypes import *
from NeuralNetwork.NeuralNetwork import *

import math
import jax.numpy as jnp
from random import choice as uniform_choice, choices as weighted_choice

type MCNode = MCNode


class MCNode():

    # Constant in u(s, a) evaluation
    _c: float

    state: AbstractState
    children: dict[Action, MCNode]

    parent: MCNode
    action_taken: Action

    visits_to_self: int
    visit_counts: dict[Action, int]
    sum_evaluation: float

    action_from_parent: Action
    
    reward: float

    def __init__(self, state: AbstractState, actions, exploration, reward: float, parent: MCNode = None, action_from_parent: Action = None):
        
        self._c = exploration
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.actions = actions
        self.children = {}
        self.visit_counts = {}
        for action in actions:
            self.visit_counts[action] = 0
        self.visits_to_self = 0
        self.sum_evaluation = 0
        self.reward = reward
        

    # Generate the children of this node.
    def expand(self, game: AbstractGame, dynamics_network: NeuralNetwork):

        assert len(self.children) == 0, 'Unexpectedly regenerated children of node.'
        
        action_space = game.action_space()

        for action in action_space:
            reward, next_abstract_state = dynamics(self.state, action, dynamics_network)
            child = MCNode(next_abstract_state, self.actions, self._c, reward, self, action)
            self.children[action] = child

    # Randomly choose a child and return it. Uniform distribution.
    # Save the action for later. It's used during backpropagation.
    def uniform_get_random_child(self) -> MCNode:
        action = uniform_choice(list(self.children.keys()))
        child = self.children[action]
        return child

    # Randomly select a child and return it.
    # Probabilities are proportional to the child's visit
    # count, so more explored children are favored. Hence,
    # is suitable to select actual action after MCTS.
    def biased_get_random_action(self) -> Action:
        # TODO: This is probably quite slow. Find better way to accomplish the same thing.
        actions = list(self.children.keys())
        weights = [self.children[action].visits_to_self for action in actions]
        chosen_action = weighted_choice(actions, weights=weights)[0]
        return chosen_action

    # A node is considered a leaf if it has no children.
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0

    # u(a) is the exploration bonus of action a
    def u(self, action: Action) -> float:
        N_sa = self.visit_counts[action]
        _u = self._c * math.sqrt(math.log2(self.visits_to_self) / (1 + N_sa))
        return _u

    # Q(a) is the value of doing action a
    def Q(self, action: Action) -> float:
        if action in self.visit_counts:
            return self.sum_evaluation / self.visit_counts[action]
        return 0.0

    # Backpropagate the value up through the tree.
    # Discount by multiplying by the discount factor (e.g. 0.95) at each step.
    def backpropagate(self, value: float, discount_factor: float):
        
        self.sum_evaluation += value
        self.visits_to_self += 1
        
        if self.action_from_parent is not None:
            if self.action_from_parent not in self.parent.visit_counts:
                self.parent.visit_counts[self.action_from_parent] = 0
            self.parent.visit_counts[self.action_from_parent] += 1

        if self.parent is None:
            return

        self.parent.backpropagate(value * discount_factor, discount_factor)
