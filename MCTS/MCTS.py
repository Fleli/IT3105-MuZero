
from random import choice as uniform_choice, choices as weighted_choice

from Game.Game import *
from MCTS.Conventions import *
from NeuralNetwork.NeuralNetwork import *

from MCTS.MCNode import *

from .Conventions import *

import numpy as np

class MCTS():

    _rollout_depth = 1
    _verbose = True

    # Actual, concrete game model (NOTE: Rename class to 'Game' since it's concrete)
    game: AbstractGame

    # (Abstract state k, Action k) -> (Abstract state k+1, Reward k)
    dynamics_network: NeuralNetwork
    prediction_network: NeuralNetwork       # (Abstract state k) -> (Value k)
    
    # (Concrete states k, k-1, ..., k-q) -> (Abstract state k)
    representation_network: NeuralNetwork

    # Initialize with Game plus three NNs
    def __init__(self, game: AbstractGame, dynamics: NeuralNetwork, prediction: NeuralNetwork, representation: NeuralNetwork, config):
        self.game = game
        self.dynamics_network = dynamics
        self.prediction_network = prediction
        self.representation_network = representation
        self.config = config
        self._rollout_depth = config['max_depth']
        self._verbose = config['verbose']

    # Do a Monte Carlo Tree Search
    # - input: A list of the (q+1) last concrete game states s_(k-q), ..., s_(k)
    # - output: The concrete move that is (hopefully) optimal
    def search(self, N_rollouts: int, concrete_game_states: jax.Array, action_space=[0,1]) -> tuple[Action, jax.Array, float]:
        
        self.log("[MCTS]")
        
        self.action_space = action_space
        
        flattened_states = concrete_game_states.flatten()
        abstract_state = self.representation_network.forward(flattened_states)
        
        root = MCNode(abstract_state, action_space, self.config['exploration'], None, None)
        
        for simulation in range(N_rollouts):
            
            self.log(f" -> [MCTS] Simulation {simulation + 1} / {N_rollouts}")
            
            # Start at root
            node = root
            self.log(f"\tnode={node.id} (root)")
            
            # Move root -> leaf, guided by tree policy
            while not node.is_leaf_node():
                action = self.tree_policy(node)
                node = node.children[action]
                self.log(f"\tnode={node.id}")
            
            self.log(f"\tfinished tree search, leaf is {node.id}")
            
            # Leaf L
            leaf = node
            leaf.expand(self.dynamics_network)
            
            # Random child c*
            action = uniform_choice(leaf.action_space)
            child = leaf.children[action]
            
            self.log(f"\trandomly chose child {child.id} (action={action})")
            
            # Do rollout
            accum_reward = self.do_rollout(child, depth=1)
            
            self.log(f"\taccum_reward={accum_reward}")
            
            # Backpropagate
            self.do_backpropagation(child, root, accum_reward)
        
        
        # -----
        # Kan vi prune treet slik at den endelige action blir ny root? Så slipper man å regenerere den delen av treet
        # neste gang. Siden denne blir valgt er den mest explored, så treet er sannsynligvis relativt tungt
        # mot denne siden.
        # -----
        
        visit_counts = [ root.visit_counts[action] for action in self.game.action_space() ]
        visit_distr = jax.nn.softmax( np.array ( visit_counts ) )
        chosen_action = weighted_choice(action_space, visit_distr)[0]
        avg_evaluation = sum(root.sum_evaluations) / N_rollouts
        
        # Get random child, probability weighted to favor those branches that are explored the most.
        results = chosen_action, visit_distr, avg_evaluation
        
        self.log("  MCTS Results:")
        self.log(f" -> Action {results[0]}")
        self.log(f" -> Visits {visit_counts}")
        self.log(f" -> Distr {results[1]}")
        self.log(f" -> Eval {results[2]}")
        
        return results
    
        
    # Do a rollout from a node (child of leaf node) to a certain depth (1).
    def do_rollout( self, node: MCNode, depth: int ) -> list[float]:
        
        accum_reward = []
        
        state = node.state
        
        for d in range(depth):
            _, policy = prediction(state, self.prediction_network)
            action = weighted_choice(self.action_space, policy)[0]
            reward, state = dynamics(state, action, self.dynamics_network)
            accum_reward.append(reward)
        
        value, _ = prediction(state, self.prediction_network)
        accum_reward[-1] = value
        
        return accum_reward
    
    
    # Do backpropagation from child of leaf node up to (root) goal.
    def do_backpropagation( self, node: MCNode, goal_node: MCNode, rewards: list[float] ):
        
        self.log(f"\t[backpropagation] rewards={rewards}")
        
        node.visits_to_self += 1
        
        # TODO: Update Q of node. Is this correct?
        if node.parent:
            node.parent.visit_counts[node.action_from_parent] += 1
            node.parent.sum_evaluations[node.action_from_parent] += jax.nn.sigmoid(sum(rewards))
            self.log(f"\tparent")
        else:
            self.log(f"\tno parent")
        
        if node != goal_node:
            self.log("\thas not reached goal")
            action = node.action_from_parent
            rewards.append( node.parent.rewards[action] )
            self.do_backpropagation(node.parent, goal_node, rewards)
        else:
            self.log("\tdone")
    
    
    # Choose the best move from a given state, evaluated by Q(s, a) + u(s, a)
    def tree_policy(self, node: MCNode) -> Action:
        value = lambda action: node.Q(action) + node.u(action, self.prediction_network)
        res = np.argmax( [ value(action) for action in self.action_space ] )
        self.log(f"\t[tree_policy] values={[ float(value(action)) for action in self.action_space ]} result={res} details={[(float(node.Q(action)),float(node.u(action,self.prediction_network))) for action in self.action_space]}")
        return res
    
    # Print a string if the verbose setting is True.
    def log(self, content: str, force=False):
        self.logger.log(content, force or self._verbose)
