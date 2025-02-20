
from Game.Game import *
from Conventions import *
from NeuralNetwork.NeuralNetwork import *

from MCNode import *

class MCTS():
    
    
    _TREE_POLICY = False
    _DEFAULT_POLICY = True
    
    
    _rollout_depth = 1
    _verbose = True
    
    
    game: AbstractGame                      # Actual, concrete game model (TODO: Rename class to 'GameInterface' or similar)
    
    dynamics_network: NeuralNetwork         # (Abstract state k, Action k) -> (Abstract state k+1, Reward k)
    prediction_network: NeuralNetwork       # (Abstract state k) -> (Value k)
    representation_network: NeuralNetwork   # (Concrete states k, k-1, ..., k-q) -> (Abstract state k)
    
    
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
                current_node = self._policy(current_node, self._TREE_POLICY)
                explored.append(current_node)
            
            current_node.expand()
            child = current_node.uniform_get_random_child()
            self._rollout(child, explored, self._rollout_depth)
        
        # Kan vi prune treet slik at den endelige action blir ny root? Så slipper man å regenerere den delen av treet
        # neste gang. Siden denne blir valgt er den mest explored, så treet er sannsynligvis relativt tungt
        # mot denne siden.
        
        # Get random child, probability weighted to favor those branches that are explored the most.
        return root.biased_get_random_action()
    
    
    def _rollout( self, leaf: MCNode, explored: list[MCNode], rollout_depth: int ):
        
        node = leaf
        
        for depth in range(rollout_depth):
            self.log(f"\t -> Rollout, depth = {depth + 1} / {rollout_depth}")
            node.expand()
            node = self._policy(node, self._DEFAULT_POLICY)
            explored.append(node)
        
        evaluation = self.prediction_network.predict(node.state)
        discount_factor = 1     # TODO: self.game.discount_factor() or similar. Function of environment and hence the game class.
        node.backpropagate(evaluation, discount_factor)
    
    
    # Run tree policy on (node, action) pair.
    def _tree_policy(self, node: MCNode, action: Action) -> tuple[float, MCNode]:
        return node.Q(action) + node.u(action), node.children[action]
    
    
    # Run default policy (dynamics + prediction network) on (node, action) pair.
    def _default_policy(self, node: MCNode, action: Action) -> tuple[float, MCNode]:
        
        # TODO: Verify that this is correct use of the networks
        dynamics_input = dynamics_network_input(node.state, action)
        next_state = self.dynamics_network.predict(dynamics_input)
        evaluation = self.prediction_network.predict(next_state)
        
        return evaluation, next_state
    
    
    # Run either the tree or default policy from a node to select one of its children.
    # Specify whether to use default or tree policy.
    def _policy(self, node: MCNode, use_default_policy: bool) -> MCNode:
        
        policy = self._default_policy if use_default_policy else self._tree_policy
        
        action_space = self.game.action_space()  # Currently returns None. TODO: PULL CHANGES FROM MAIN, RESOLVE BEFORE PUSH.
        best_next = None
        best_evaluation = 0.0  # NOTE: Make sure evaluations are in [0 , 1], which is assumed here.
        
        for action in action_space:
            evaluation, next = policy(node, action)
            if evaluation > best_evaluation:
                best_next = next
                best_evaluation = evaluation
        
        return best_next
    
    
    def log(self, content: str):
        if self._verbose:
            print(content)
    