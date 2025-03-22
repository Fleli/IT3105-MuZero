
from random import choices as weighted_choice

from Game.Game import *
from MCTS.Conventions import *
from NeuralNetwork.NeuralNetwork import *

from MCTS.MCNode import *

import jax.numpy as jnp

ID = 0

class MCTS():
    
    id: int
    
    _rollout_depth = 1
    _verbose = True
    
    game: AbstractGame                      # Actual, concrete game model (NOTE: Rename class to 'Game' since it's concrete)
    
    dynamics_network: NeuralNetwork         # (Abstract state k, Action k) -> (Abstract state k+1, Reward k)
    prediction_network: NeuralNetwork       # (Abstract state k) -> (Value k)
    representation_network: NeuralNetwork   # (Concrete states k, k-1, ..., k-q) -> (Abstract state k)
    
    # Initialize with Game plus three NNs
    def __init__(self, game: AbstractGame, dynamics: NeuralNetwork, prediction: NeuralNetwork, representation: NeuralNetwork):
        self.game = game
        self.dynamics_network = dynamics
        self.prediction_network = prediction
        self.representation_network = representation
        
        global ID
        self.id = ID
        ID += 1
    
    
    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal
    def search(self, N_rollouts: int, concrete_game_states: jax.Array) -> tuple[Action, dict, float]:
        
        self.log("Search for next actual move")
        
        print("Concrete game states:", concrete_game_states)
        flattened_states = concrete_game_states.flatten()
        print("Flattened states:", flattened_states)
        abstract_state, _representation_hidden = self.representation_network.predict(flattened_states)
        
        print("Abstract state returned:", abstract_state)
        
        root = MCNode(abstract_state, None, None)
        
        print(f"Address of root:", root.__str__())
        
        for simulation in range(N_rollouts):
            
            self.log(f" -> Simulation {simulation + 1} / {N_rollouts}")
            
            current_node = root
            
            while not current_node.is_leaf_node(): 
                print(f"[in while loop] current_node={current_node.__str__()}")
                print(f"[children]: {[value.__str__() for key, value in current_node.children.items()]}")
                action = self._tree_policy(current_node)
                print(f"action={action}")
                current_node = current_node.children[action]
                print(f"[children after reassignment]: {[value.__str__() for key, value in current_node.children.items()]}")
            
            print(f"{self.search.__name__}: Will expand current node (finished tree policy down to leaf)")
            self._rollout(current_node, self._rollout_depth)
        
        # Kan vi prune treet slik at den endelige action blir ny root? Så slipper man å regenerere den delen av treet
        # neste gang. Siden denne blir valgt er den mest explored, så treet er sannsynligvis relativt tungt
        # mot denne siden.
        
        # Get random child, probability weighted to favor those branches that are explored the most.
        return root.biased_get_random_action(), root.visit_counts, root.sum_evaluation
    
    
    # Do a rollout to a certain depth, and backpropagate the result afterwards.
    def _rollout(self, leaf: MCNode, rollout_depth: int):
        
        print(f" [in rollout] leaf={leaf}")
        print(f" [----------] children={leaf.children}")
        
        node = leaf
        
        for depth in range(rollout_depth):
            self.log(f"\t -> Rollout, depth = {depth + 1} / {rollout_depth}")
            node.expand(self.game, self.dynamics_network)
            action = self._default_policy(node)
            node = node.children[action]
        
        print(f"{self._rollout.__name__}:")
        print("\tWill find evaluation")
        print(f"\tnode.state={node.state}")
        
        # Evaluate the leaf state, but throw away the action probabilities (they're irrelevant here).
        nn_output, _prediction_hidden = self.prediction_network.predict(node.state)
        evaluation, _ = prediction_network_output(nn_output)
        discount_factor = 1     # TODO: self.game.discount_factor() or similar. Function of environment and hence the game class.
        node.backpropagate(evaluation, discount_factor)
    
    
    # Choose the best move from a given state, evaluated by Q(s, a) + u(s, a)
    def _tree_policy(self, node: MCNode) -> Action:
        action_space = self.game.action_space()
        best_action = None
        best_evaluation = 0.0  # NOTE: Make sure evaluations are in [0 , 1], which is assumed here.
        for action in action_space:
            print(f"\n\t\t -> action={action}")
            evaluation = node.Q(action) + node.u(action)
            print(f"\t\t -> evaluation={evaluation}")
            if evaluation > best_evaluation:
                best_action = action
                best_evaluation = evaluation
        return best_action
    
    
    # Choose a random move with weighted probabilities using the prediction network.
    def _default_policy(self, node: MCNode) -> Action:
        print(f"{self._default_policy.__name__}:")
        action_space = self.game.action_space()    # List of actions? Need to agree on interface here.
        print(f"\taction_space={action_space}")
        prediction, _prediction_hidden = self.prediction_network.predict(node.state)
        print(f"\tprediction={prediction}")
        _, probabilities = prediction_network_output(prediction)
        print(f"\tprobabilities={probabilities}")
        return weighted_choice(action_space, probabilities)[0]
    
    
    # Print a string if the verbose setting is True.
    def log(self, content: str):
        if self._verbose:
            print(content)
            
            
    def __str__(self):
        return str(self.id)
    
    