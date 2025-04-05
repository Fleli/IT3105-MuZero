
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

    action_from_parent: Action = None

    def __init__(self, state: AbstractState, actions, exploration, reward=None, parent: MCNode = None, action_from_parent: Action = None):
        if reward is None:
            reward = jnp.array([1])
        self._c = exploration
        self.state = state
        self.reward = reward
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.actions = actions

        self.children = {}
        self.visit_counts = {}
        for action in actions:
            self.visit_counts[action] = 0
        self.visits_to_self = 0
        self.sum_evaluation = 0

    # Generate the children of this node.

    def expand(self, game: AbstractGame, dynamics_network: NeuralNetwork):

        assert len(
            self.children) == 0, 'Unexpectedly regenerated children of node.'
        action_space = game.action_space()

        for action in action_space:
            network_input = dynamics_network_input(self.state, action)
            nn_output = dynamics_network.forward(network_input)
            reward, next_abstract_state = dynamics_network_output(nn_output)
            reward = jnp.array([reward])
            child = MCNode(next_abstract_state, self.actions,
                           self._c, reward, self, action)
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
        return weighted_choice(actions, weights=weights)[0]

    # A node is considered a leaf if it has no children.

    def is_leaf_node(self) -> bool:
        return len(self.children) == 0

    # u(a) is the exploration bonus of action a

    def u(self, action: Action) -> float:
        N_sa = self.visit_counts[action] if action in self.visit_counts else 0
        return self._c * math.sqrt(math.log2(self.visits_to_self) / (1 + N_sa))

    # Q(a) is the value of doing action a

    def Q(self, action: Action) -> float:
        child = self.children.get(action)
        if child and child.visits_to_self > 0:
            return child.sum_evaluation / child.visits_to_self
        return 0

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
