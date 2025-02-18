
from NeuralNetwork.NeuralNetwork import *

type ConcreteGameState = list[int]

class MCTS:
    
    tree_policy: NeuralNetwork
    default_policy: NeuralNetwork
    
    def __init__(self):
        pass
    
    
    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal
    def search( concrete_game_states: list[ConcreteGameState] ):
        
        
        
        return None  # TODO: Return 
    
    
    def _rollout(self):
        pass
    
    def _backpropagate(self):
        pass
    