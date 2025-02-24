
from random import choices as weighted_choice

from Game.Game import *
from Conventions import *
from NeuralNetwork.NeuralNetwork import *

from MCNode import *

class MCTS():
    
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
    
    
    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal
    def search( self, N_rollouts: int, concrete_game_states: list[ConcreteGameState] ) -> Action:
        
        self.log("Search for next actual move")
        
        abstract_state: AbstractState = self.representation_network.predict(concrete_game_states)
        
        root = MCNode(abstract_state, None)
        
        for simulation in range(N_rollouts):
            
            self.log(f" -> Simulation {simulation + 1} / {N_rollouts}")
            
            current_node = root
            explored = [root]
            
            while not current_node.is_leaf_node(): 
                action = self._tree_policy(current_node)
                current_node = current_node.children[action]
                explored.append(current_node)
            
            current_node.expand()
            child = current_node.uniform_get_random_child()
            self._rollout(child, explored, self._rollout_depth)
        
        # Kan vi prune treet slik at den endelige action blir ny root? Så slipper man å regenerere den delen av treet
        # neste gang. Siden denne blir valgt er den mest explored, så treet er sannsynligvis relativt tungt
        # mot denne siden.
        
        # Get random child, probability weighted to favor those branches that are explored the most.
        return root.biased_get_random_action()
    
    
    # Do a rollout to a certain depth, and backpropagate the result afterwards.
    def _rollout(self, leaf: MCNode, explored: list[MCNode], rollout_depth: int):
        
        node = leaf
        
        for depth in range(rollout_depth):
            self.log(f"\t -> Rollout, depth = {depth + 1} / {rollout_depth}")
            node.expand()
            action = self._default_policy(node)
            node = node.children[action]
            explored.append(node)
        
        # Evaluate the leaf state, but throw away the action probabilities (they're irrelevant here).
        evaluation, _ = prediction_network_output(self.prediction_network.predict(node.state))
        discount_factor = 1     # TODO: self.game.discount_factor() or similar. Function of environment and hence the game class.
        node.backpropagate(evaluation, discount_factor)
    
    
    # Choose the best move from a given state, evaluated by Q(s, a) + u(s, a)
    def _tree_policy(self, node: MCNode) -> Action:
        action_space = self.game.action_space()
        best_action = None
        best_evaluation = 0.0  # NOTE: Make sure evaluations are in [0 , 1], which is assumed here.
        for action in action_space:
            evaluation = node.Q(action) + node.u(action)
            if evaluation > best_evaluation:
                best_action = action
                best_evaluation = evaluation
        return best_action
    
    
    # Choose a random move with weighted probabilities using the prediction network.
    def _default_policy(self, node: MCNode) -> Action:
        action_space: list[Action] = self.game.action_space()    # List of actions? Need to agree on interface here.
        probabilities = prediction_network_output(self.prediction_network.predict(node.state))
        return weighted_choice(action_space, probabilities)[0]
    
    
    # Print a string if the verbose setting is True.
    def log(self, content: str):
        if self._verbose:
            print(content)
    