import random
from Config import CONFIG
from MCTS import MCTS
from NeuralNetwork import NeuralNetwork

class System:
    def __init__(self):
        self.Ne = CONFIG["num_episodes"]
        self.Nes = CONFIG["num_episode_steps"]
        self.Ms = CONFIG["num_searches"]
        self.dmax = CONFIG["max_depth"]
        self.It = CONFIG["training_interval"]
        self.mbs = CONFIG["minibatch_size"]
        self.game_type = CONFIG["game"]
        self.dynamic = NeuralNetwork(CONFIG["dynamics_nn"])
        self.prediction = NeuralNetwork(CONFIG["prediction_nn"])
        self.representation = NeuralNetwork(CONFIG["representation_nn"])
        self.game = self.initialize_game()
        self.mcts = MCTS(self.game, self.dynamic, self.prediction, self.representation)
        self.EH = []

    def initialize_game(self):
        """Initialize the game environment."""
        # May only have one game
        if self.game_type == "gym":
            self.game = Gym(CONFIG["flappy_bird"])


    def train(self):
        """Main training loop over episodes."""

        for episode in range(self.Ne):
            epidata = self.episode()
            self.EH.append(epidata)

            if episode % self.It == 0:
                self.do_bptt_training(self.EH, self.mbs)

    def episode(self):
        """Run a single episode and return collected data."""
        state = self.game.reset()
        epidata = []

        for k in range(self.Nes):
            step_data = self.step(state, k)
            epidata.append(step_data)
            state = step_data[0]

        return epidata
    
    def step(self, state, k):
        """Perform one step in the episode, returning collected data."""
        phi_k = self.game.gather_states(state, k)
        sigma_k = self.nn.NNr(phi_k)
        tree_root = self.mcst.initialize_tree(sigma_k)
        
        pi_k, v_k = self.search(tree_root)
        action_k = self.mcst.sample_action(pi_k)
        next_state, next_reward = self.game.simulate(state, action_k)
        
        return [state, v_k, pi_k, action_k, next_reward]
    
    def search(self, tree_root):
        """Perform tree search and return action distribution and value estimate."""
        for m in range(self.Ms):
            leaf = self.mcst.tree_policy(tree_root)
            self.mcst.expand_tree(leaf)

            if leaf.children: # If leaf is not terminal
                c_star = random.choice(leaf.children)  # Choose a random child node
                accum_reward = self.mcst.rollout(c_star, self.dmax - c_star.depth, self.nn)
                self.mcst.backpropagate(c_star, tree_root, accum_reward)
        
        pi_k = self.mcst.get_visit_distribution(tree_root)
        v_k = self.mcst.get_root_value(tree_root)
        return pi_k, v_k

    def do_bptt_training(self):
        """Perform BPTT training with the episode history."""
        self.w = self.game.w
        self.q = self.game.q

        for _ in range(self.mbs):
            Eb = random.choice(self.EH)
            k = random.randint(self.q, len(Eb) - self.w - 1)
            
            states = [Eb[i][0] for i in range(k - self.q, k + 1)]
            actions = [Eb[i][3] for i in range(k + 1, k + self.w + 1)]
            policies = [Eb[i][2] for i in range(k, k + self.w + 1)]
            values = [Eb[i][1] for i in range(k, k + self.w + 1)]
            rewards = [Eb[i][4] for i in range(k + 1, k + self.w + 1)]
            
            self.nn.do_bptt(states, actions, policies, values, rewards)
