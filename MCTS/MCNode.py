
from Game.Game import *
from MCTS.Conventions import *
from MCTS.MCTSTypes import *
from NeuralNetwork.NeuralNetwork import *

from math import sqrt, log

type MCNode = MCNode

class MCNode():
    
    # Constant in u(s, a) evaluation
    _c: float
    
    state: AbstractState
    children: list[MCNode]
    
    parent: MCNode
    action_taken: Action
    
    visits_to_self: int
    
    action_from_parent: Action
    
    rewards: list[float]
    visit_counts: list[int]
    sum_evaluations: list[float]
    
    def __init__(self, state: AbstractState, action_space, exploration, parent: MCNode = None, action_from_parent: Action = None):
        
        self._c = exploration
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.action_space = action_space
        self.visits_to_self = 0
        
        self.children = []
        
        self.rewards = []
        self.visit_counts = []
        self.sum_evaluations = []
        
        for _ in range(len(self.action_space)):
            self.rewards.append(0)
            self.visit_counts.append(0)
            self.sum_evaluations.append(0)
    
    
    # Generate the children of this node.
    def expand(self, dynamics_network: NeuralNetwork):

        assert len(self.children) == 0, 'Unexpectedly regenerated children of node.'
        
        for action in self.action_space:
            reward, next_abstract_state = dynamics(self.state, action, dynamics_network)
            child = MCNode(next_abstract_state, self.action_space, self._c, self, action)
            self.rewards[action] = reward
            self.children.append(child)
    
    
    # A node is considered a leaf if it has no children.
    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    
    # u(a) is the exploration bonus of action a
    def u(self, action: Action, pred_network) -> float:
        
        if False:
            N_sa = self.visit_counts[action]
            _u = self._c * math.sqrt(math.log2(self.visits_to_self) / (1 + N_sa))
            return _u
        
        
        # c = self._c
        
        c1 = 1.25
        c2 = 19_652
        
        _, policy = prediction(self.state, pred_network)
        p_sa = policy[action]
        N_s = self.visits_to_self
        N_sa = self.visit_counts[action]
        frac = (N_s + c2 + 1) / c2
        
        return p_sa * sqrt(N_s) / (1 + N_sa) * (c1 + log(frac))
    
    
    # Q(a) is the value of doing action a
    def Q(self, action: Action) -> float:
        if self.visit_counts[action] > 0:
            return self.sum_evaluations[action] / self.visit_counts[action]
        return 0.0
