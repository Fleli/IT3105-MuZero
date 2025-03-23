import random
from Config import CONFIG
from MCTS import MCTS
from NeuralNetwork import NeuralNetwork

import random
import gym
import jax
import jax.numpy as jnp
from Config import CONFIG

import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

class GymGame:
    
    def __init__(self, config):
        self.state_history = []
        self.w = config["w"]
        self.q = config["q"]
        self.env = gym.make(CONFIG['gym']["env_name"], render_mode="human")
    
    
    def reset(self):
        
        self.state_history = []
        result = self.env.reset()
        
        if isinstance(result, tuple):
            observation, _ = result
        else:
            observation = result
        
        return observation
    
    
    def simulate(self, state, action):
        
        result = self.env.step(action)
        
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
        else:
            next_state, reward, terminated, info = result
        
        self.terminated = terminated or (len(result) == 5 and truncated)
        
        print(self.terminated)
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        self.state_history.append(next_state)
        
        return next_state, reward
    
    
    def gather_states(self, state, k):
        
        states = []
        
        for state_index in range(k - self.q, k):
            if state_index <= 0:
                states.append(jnp.zeros_like(state))
            else:
                states.append(self.state_history[state_index])
        
        return jnp.array(states + [state])
    
    
    def action_space(self):
        return list(range(self.env.action_space.n))
    
    
    def render(self):
        self.env.render()



class System:
    
    def __init__(self):
        self.num_episodes = CONFIG["num_episodes"]
        self.num_episode_steps = CONFIG["num_episode_steps"]
        self.num_searches = CONFIG["num_searches"]
        self.dmax = CONFIG["max_depth"]
        self.training_int = CONFIG["training_interval"]
        self.mini_batch_size = CONFIG["minibatch_size"]
        self.game_type = CONFIG["game"]
        self.dynamics = NeuralNetwork(CONFIG["dynamics_nn"])
        self.prediction = NeuralNetwork(CONFIG["prediction_nn"])
        self.representation = NeuralNetwork(CONFIG["representation_nn"])
        self.game = self.initialize_game()
        self.mcts = MCTS(self.game, self.dynamics, self.prediction, self.representation)
        self.epidata_array = []
    
    
    def initialize_game(self):
        """Initialize the game environment."""
        # May only have one game
        if self.game_type == "gym":
            game = GymGame(CONFIG["gym"])

        return game
    
    
    def train(self):
        """Main training loop over episodes."""
        for episode in range(self.num_episodes):
            epidata = self.episode()
            self.epidata_array.append(epidata)
            
            if episode % self.training_int == 0:
                # self.do_bptt_training(self.EH, self.mbs)      # Feil call-signatur
                self.do_bptt_training()
    
    
    def episode(self):
        """Run a single episode and return collected data."""
        state = self.game.reset()
        epidata = []
        
        print(f"episode. Note self.Nes={self.num_episode_steps}")
        
        for k in range(self.num_episode_steps):
            step_data = self.step(state, k)
            epidata.append(step_data)
            state = step_data[0]

        return epidata
    
    
    def step(self, state, k):
        """Perform one step in the episode, returning collected data."""
        phi_k = self.game.gather_states(state, k)

        action_k, visit_dist, root_value = self.mcts.search(self.num_searches, phi_k)
        next_state, next_reward = self.game.simulate(state, action_k)
        
        # TODO: Det sto [state, ...] her. Har endret til [next_state, ...] fordi vi skal vel ha den neste staten
        # til neste step? Dobbeltsjekk.
        return [next_state, root_value, visit_dist, action_k, next_reward]
    
    
    def do_bptt_training(self):
        """Perform BPTT training with the episode history."""
        self.w = self.game.w
        self.q = self.game.q

        for _ in range(self.mini_batch_size):
            random_epidata = random.choice(self.epidata_array)
            k = random.randint(self.q, len(random_epidata) - self.w - 1)
            
            states = [random_epidata[i][0] for i in range(k - self.q, k + 1)]
            actions = [random_epidata[i][3] for i in range(k + 1, k + self.w + 1)]
            policies = [random_epidata[i][2] for i in range(k, k + self.w + 1)]
            values = [random_epidata[i][1] for i in range(k, k + self.w + 1)]
            rewards = [random_epidata[i][4] for i in range(k + 1, k + self.w + 1)]

            PVR = [policies, values, rewards]
            
            self.representation.BPTT(states, actions, PVR)
            self.prediction.BPTT(states, actions, PVR)
            self.dynamics.BPTT(states, actions, PVR)


if __name__ == '__main__':
    
    system = System()
    system.train()
