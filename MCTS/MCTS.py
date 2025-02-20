
from NeuralNetwork.NeuralNetwork import *

from MCNode import *

type ConcreteGameState = list[float]

class MCTS():
    
    
    representation_network: NeuralNetwork
    
    
    def __init__(self, representation_network: NeuralNetwork):
        self.representation_network = representation_network
        
    
    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal
    def search( self, N_rollouts: int, concrete_game_states: list[ConcreteGameState] ):
        
        abstract_state: AbstractState = self.representation_network.predict(concrete_game_states)
        
        root = MCNode(abstract_state)
        
        for _ in range(N_rollouts):
            
            current_node = root
            explored = [root]
            
            while not current_node.is_leaf_node(): 
                current_node = self._tree_policy(current_node)
                explored.append(current_node)
            
            current_node.expand()
            child = current_node.get_random_child()
            terminal_value = self._rollout(child, explored)       # Legg til variabel lengde på rollout?
            
        
        return None  # TODO: Return 
    
    
    def _tree_policy(self, node: MCNode) -> MCNode:
        return None  # TODO: Implement
    
    
    def _rollout( self, leaf: MCNode, explored: list[MCNode] ) -> float:
        
        for node in explored:
            pass
        
        return 0
    
    
    def _backpropagate(self):
        pass
    