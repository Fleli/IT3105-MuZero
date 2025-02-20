
from NeuralNetwork.NeuralNetwork import *

from MCNode import *

class MCTS():
    
    
    representation_network: NeuralNetwork
    
    
    def __init__(self, representation_network: NeuralNetwork):
        self.representation_network = representation_network
        
    
    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal
    def search( self, N_rollouts: int, concrete_game_states: list[ConcreteGameState] ) -> Action:
        
        abstract_state: AbstractState = self.representation_network.predict(concrete_game_states)
        
        root = MCNode(abstract_state)
        
        for _ in range(N_rollouts):
            
            current_node = root
            explored = [root]
            
            while not current_node.is_leaf_node(): 
                current_node = self._tree_policy(current_node)
                explored.append(current_node)
            
            current_node.expand()
            child = current_node.uniform_get_random_child()
            terminal_value = self._rollout(child, explored)       # Legg til variabel lengde på rollout?
        
        
        
        # Kan vi prune treet slik at den endelige action blir ny root? Så slipper man å regenerere den delen av treet
        # neste gang. Siden denne blir valgt er den mest explored, så treet er sannsynligvis relativt tungt
        # mot denne siden.
        
        
        
        
        # Get random child, probability weighted to favor those branches that are explored the most.
        return root.biased_get_random_action()
    
    
    def _tree_policy(self, node: MCNode) -> MCNode:
        return None  # TODO: Implement
    
    
    def _rollout( self, leaf: MCNode, explored: list[MCNode] ) -> float:
        
        for node in explored:
            pass
        
        return 0
    
    
    def _backpropagate(self):
        pass
    