
from Game.Game import *
from MCTS.Conventions import *
from NeuralNetwork.NeuralNetwork import *

from MCTS.MCNode import *

from .Conventions import *

import jax.numpy as jnp
import jax.random as jrandom

import numpy as np


class MCTS():

    _rollout_depth = 1
    _verbose = False

    # Actual, concrete game model (NOTE: Rename class to 'Game' since it's concrete)
    game: AbstractGame

    # (Abstract state k, Action k) -> (Abstract state k+1, Reward k)
    dynamics_network: NeuralNetwork
    prediction_network: NeuralNetwork       # (Abstract state k) -> (Value k)
    # (Concrete states k, k-1, ..., k-q) -> (Abstract state k)
    representation_network: NeuralNetwork

    # Initialize with Game plus three NNs

    def __init__(self, game: AbstractGame, dynamics: NeuralNetwork, prediction: NeuralNetwork, representation: NeuralNetwork, config, key):
        self.game = game
        self.dynamics_network = dynamics
        self.prediction_network = prediction
        self.representation_network = representation
        self.config = config
        self._rollout_depth = config['max_depth']
        self._verbose = config['verbose']
        self.key = key
        
    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal

    def search(self, N_rollouts: int, concrete_game_states: jax.Array, actions=[0,1]) -> tuple[Action, dict, float]:
        
        self.log(f"Starting new search: N_rollouts={N_rollouts}, actions={actions}", force=True)
        self.log(f"concrete_game_states={concrete_game_states}", force=True)
        
        flattened_states = concrete_game_states.flatten()
        abstract_state = self.representation_network.forward(flattened_states)

        self.log(f"Concrete game states ( \n{concrete_game_states} )\n")
        self.log(f"Flattened states:{flattened_states}")
        self.log(f"Abstract state returned:{abstract_state}")
        root = MCNode(abstract_state, actions, self.config['exploration'], None, None, None)

        for simulation in range(N_rollouts):

            self.log(f" -> Simulation {simulation + 1} / {N_rollouts}", force=True)

            current_node = root
            while not current_node.is_leaf_node():
                if self._verbose:
                    print(
                        f"[in while] current={current_node.__hash__()}, children={[f"{child.__hash__()}" for action, child in current_node.children.items()]}")
                action = self._tree_policy(current_node)
                current_node = current_node.children[action]
            
            self.log(f"\t\tFinished tree policy-guided move to leaf. Expanding current node.", force=True)
            current_node.expand(self.game, self.dynamics_network)
            
            self.log(f"\t\tcurrent_node.children.keys()={list(current_node.children.keys())}", force=True)
            self.key, subkey = jrandom.split(self.key)
            random_action = int( jrandom.choice(subkey, jnp.array(list(current_node.children.keys())) ) )
            random_child = current_node.children[random_action]
            self.log(f"\t\trandom action for rollout: {random_action}", force=True)
            self._rollout(random_child, self._rollout_depth)

        # Kan vi prune treet slik at den endelige action blir ny root? Så slipper man å regenerere den delen av treet
        # neste gang. Siden denne blir valgt er den mest explored, så treet er sannsynligvis relativt tungt
        # mot denne siden.
        
        sum_visits = sum(root.visit_counts.values())
        visit_distr = jax.nn.softmax(np.array([ root.visit_counts[action] for action in self.game.action_space() ]))
        # visit_distr = { action: root.visit_counts[action] / sum_visits for action in self.game.action_space() }

        # Get random child, probability weighted to favor those branches that are explored the most.
        self.key, action = root.biased_get_random_action(self.key)
        results = action, visit_distr, root.sum_evaluation/N_rollouts
        
        self.log("MCTS Results:", force=True)
        self.log(f" -> Action {results[0]}", force=True)
        self.log(f" -> Visits {results[1]}", force=True)
        self.log(f" -> Eval {results[2]}", force=True)

        return results

    # Do a rollout to a certain depth, and backpropagate the result afterwards.

    def _rollout(self, leaf: MCNode, rollout_depth: int):
        
        node = leaf
        for depth in range(rollout_depth):
            self.log(f"\t -> Rollout, depth = {depth + 1} / {rollout_depth}", force=True)
            node.expand(self.game, self.dynamics_network)
            action = self._default_policy(node)
            node = node.children[action]

        # Evaluate the leaf state, but throw away the action probabilities (they're irrelevant here).
        evaluation, _ = prediction(node.state, self.prediction_network)
        discount_factor = self.config['discount_factor']
        node.backpropagate(evaluation, discount_factor)

    # Choose the best move from a given state, evaluated by Q(s, a) + u(s, a)

    def _tree_policy(self, node: MCNode) -> Action:
        
        self.log(f"\t\t[tree policy]", force=True)
        action_space = self.game.action_space()
        best_action = None
        # NOTE: Make sure evaluations are in [0 , 1], which is assumed here.
        best_evaluation = -100
        for action in action_space:
            evaluation = node.Q(action) + node.u(action)
            self.log(f"\t\t -> evaluation={evaluation}", force=True)
            self.log(f"\t\t -> action={action}", force=True)
            if evaluation > best_evaluation:
                best_action = action
                best_evaluation = evaluation
        self.log(f"\t\treturned from tree policy: {best_action}", force=True)
        return best_action

    # Choose a random move with weighted probabilities using the prediction network.

    def _default_policy(self, node: MCNode) -> Action:
        # List of actions? Need to agree on interface here.
        action_space = self.game.action_space()
        _, probabilities = prediction(node.state, self.prediction_network)
        self.key, action = weighted_choice(probabilities, action_space, self.key)
        self.log(f"\t\t\t[default policy] probabilities={probabilities}", force=True)
        self.log(f"\t\t\t[default policy] action={action}", force=True)
        return action

    # Print a string if the verbose setting is True.

    def log(self, content: str, force=False):
        self.logger.log(content, force or self._verbose)