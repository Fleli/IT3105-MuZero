
from AbstractState import *

import random
import math

type MCNode = MCNode
type Action = int

class MCNode():
    
    # Constant in u(s, a) evaluation
    _c = 1
    
    state: AbstractState
    children: list[MCNode] = []
    
    parent: MCNode
    action_taken: Action
    
    visits_to_self = 0
    visit_counts: dict[Action, int] = {}
    sum_evaluation: float = 0
    
    
    def __init__(self, state: AbstractState):
        self.state = state
    
    
    # TODO: Generate the children of this node.
    def expand(self):
        pass
    
    
    # Randomly choose a child and return it
    def get_random_child(self) -> MCNode:
        # Also set 'action_taken'
        return random.choice(self.children)
    
    
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    
    def u(self, action: Action) -> float:
        N_sa = self.visit_counts[action]
        return self._c * math.sqrt( math.log2(self.visits_to_self) / (1 + N_sa) )
    
    
    def Q(self, action: Action) -> float:
        return self.visit_counts[action] / self.sum_evaluation
    
    
    def backpropagate(self, value: float):
        self.sum_evaluation += value
        self.visits_to_self += 1
        
        if self.parent == None:
            return
        
        self.parent.backpropagate(value)
    