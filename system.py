import random
from Config import CONFIG
from MCTS import MCTS
from NeuralNetwork import NeuralNetwork

class System:
    def __init__(self):
        self.Ne = CONFIG["num_episodes"]
        self.Nes = CONFIG["num_episode_steps"]
        self.num_searches = CONFIG["num_searches"]
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
            game = Gym(CONFIG["flappy_bird"])

        return game


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

        action_k, visit_dist, root_value = self.mcts.search(self.num_searches, phi_k)
        next_state, next_reward = self.game.simulate(state, action_k)

        return [state, root_value, visit_dist, action_k, next_reward]

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
