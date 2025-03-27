
from random import choices as weighted_choice

from Game.Game import *
from MCTS.Conventions import *
from NeuralNetwork.NeuralNetwork import *

from MCTS.MCNode import *


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

    def __init__(self, game: AbstractGame, dynamics: NeuralNetwork, prediction: NeuralNetwork, representation: NeuralNetwork):
        self.game = game
        self.dynamics_network = dynamics
        self.prediction_network = prediction
        self.representation_network = representation

    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal

    def search(self, N_rollouts: int, concrete_game_states: jax.Array, actions=[0,1]) -> tuple[Action, dict, float]:

        flattened_states = concrete_game_states.flatten()
        abstract_state = self.representation_network.forward(flattened_states)

        self.log(f"Concrete game states ( \n{concrete_game_states} )\n")
        self.log(f"Flattened states:{flattened_states}")
        self.log(f"Abstract state returned:{abstract_state}")
        root = MCNode(abstract_state, actions, None, None, None)

        for simulation in range(N_rollouts):

            self.log(
                f" -> Simulation {simulation + 1} / {N_rollouts}")

            current_node = root

            while not current_node.is_leaf_node():
                if self._verbose:
                    print(
                        f"[in while] current={current_node.__hash__()}, children={[f"{child.__hash__()}" for action, child in current_node.children.items()]}")
                action = self._tree_policy(current_node)
                current_node = current_node.children[action]

            self._rollout(current_node, self._rollout_depth)

        # Kan vi prune treet slik at den endelige action blir ny root? Så slipper man å regenerere den delen av treet
        # neste gang. Siden denne blir valgt er den mest explored, så treet er sannsynligvis relativt tungt
        # mot denne siden.

        # Get random child, probability weighted to favor those branches that are explored the most.
        results = root.biased_get_random_action(), root.visit_counts, root.sum_evaluation/N_rollouts

        self.log("MCTS Results:", force=True)
        self.log(f" -> Action {results[0]}", force=True)
        self.log(f" -> Visits {results[1]}", force=True)
        self.log(f" -> Eval {results[2]}", force=True)

        return results

    # Do a rollout to a certain depth, and backpropagate the result afterwards.

    def _rollout(self, leaf: MCNode, rollout_depth: int):

        node = leaf
        for depth in range(rollout_depth):
            self.log(f"\t -> Rollout, depth = {depth + 1} / {rollout_depth}")
            node.expand(self.game, self.dynamics_network)
            action = self._default_policy(
                node)
            node = node.children[action]

        # Evaluate the leaf state, but throw away the action probabilities (they're irrelevant here).
        nn_output = self.prediction_network.forward(
            jnp.concatenate([node.reward, node.state]))
        evaluation, _ = prediction_network_output(nn_output)
        # TODO: self.game.discount_factor() or similar. Function of environment and hence the game class.
        discount_factor = 0.95
        node.backpropagate(evaluation, discount_factor)

    # Choose the best move from a given state, evaluated by Q(s, a) + u(s, a)

    def _tree_policy(self, node: MCNode) -> Action:
        action_space = self.game.action_space()
        best_action = None
        # NOTE: Make sure evaluations are in [0 , 1], which is assumed here.
        best_evaluation = 0.0
        for action in action_space:
            evaluation = node.Q(action) + node.u(action)

            self.log(f"\t\t -> evaluation={evaluation}")
            self.log(f"\n\t\t -> action={action}")

            if evaluation > best_evaluation:
                best_action = action
                best_evaluation = evaluation
        return best_action

    # Choose a random move with weighted probabilities using the prediction network.

    def _default_policy(self, node: MCNode) -> Action:
        # List of actions? Need to agree on interface here.
        action_space = self.game.action_space()
        prediction = self.prediction_network.forward(
            jnp.concatenate([node.reward, node.state]))
        _, probabilities = prediction_network_output(prediction)
        return weighted_choice(action_space, probabilities)[0]

    # Print a string if the verbose setting is True.

    def log(self, content: str, force=False):
        if self._verbose or force:
            print(content)
