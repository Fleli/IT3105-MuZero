import random
from Config import CONFIG
from MCTS import MCTS

from NeuralNetworkManager import *

from utils import Logger

import random
import gym
import jax.numpy as jnp
# TODO: Må MCTS.game resettes mellom hver epoch?

import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


def to_float(value):
    if isinstance(value, jnp.ndarray):
        try:
            return float(value)
        except Exception:
            return float(value.tolist()[0])
    return value


class GymGame:

    def __init__(self, config):
        self.state_history = []
        self.roll_ahead = config["w"]
        self.look_back = config["q"]
        self.env = gym.make(config["env_name"], render_mode="human")

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

        if isinstance(next_state, tuple):
            next_state = next_state[0]

        self.state_history.append(next_state)

        return next_state, reward

    def gather_states(self, state, k):

        states = []

        for state_index in range(k - self.look_back, k):
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

    def __init__(self, config):
        self.num_episodes = config["num_episodes"]
        self.num_episode_steps = config["num_episode_steps"]
        self.num_searches = config["num_searches"]
        self.dmax = config["max_depth"]
        self.training_int = config["training_interval"]
        self.mini_batch_size = config["minibatch_size"]
        self.game_type = config["game"]
        self.nnm = NeuralNetworkManager(config)
        self.game = self.initialize_game(config['gym'])
        self.discount_factor = config["discount_factor"]
        self.mcts = MCTS(self.game, self.nnm.dynamics,
                         self.nnm.prediction, self.nnm.representation, config)
        self.epidata_array = []

    def initialize_game(self, gym_params):
        """Initialize the game environment."""
        # May only have one game
        if self.game_type == "gym":
            game = GymGame(gym_params)

        return game

    def train(self):
        
        print(self.game.action_space())
        
        """Main training loop over episodes."""
        logger = Logger()
        self.mcts.logger = logger
        score = 0
        last_score = 0
        for episode in range(self.num_episodes):
            epidata = self.episode()
            self.epidata_array.append(epidata)
            score += len(epidata)
            if episode % self.training_int == 0:
                self.do_bptt_training()
                print("="*60, score/self.training_int)
                self.mcts.log("="*60+f" {score/self.training_int}")
                last_score = score
                score = 0

        return last_score/self.training_int

    def episode(self):
        """Run a single episode and return collected data."""
        self.mcts.log("--episode--", force=True)
        state = self.game.reset()
        epidata = []
        reward = 1
        # print(f"episode. Note self.Nes={self.num_episode_steps}")
        for k in range(self.num_episode_steps):
            step_data = self.step(state, k, reward)
            state, reward, step_data[0], step_data[4] = step_data[0], step_data[4], state, reward
            epidata.append(step_data)
            
            self.mcts.log(f"k={k}", force=True)
            self.mcts.log(f"step_data={step_data}", force=True)
            
            if self.game.terminated:
                break
        self.mcts.log(f"Value current episode: {len(epidata)}", force=True)
        print("Value current episode:", len(epidata))
        return epidata

    def step(self, state, k, reward):
        """Perform one step in the episode, returning collected data."""
        phi_k = self.game.gather_states(state, k)
        action_k, visit_dist, root_value = self.mcts.search(self.num_searches, phi_k)
        next_state, next_reward = self.game.simulate(state, action_k)

        # TODO: Det sto [state, ...] her. Har endret til [next_state, ...] fordi vi skal vel ha den neste staten
        # til neste step? Dobbeltsjekk.
        return [next_state, root_value, visit_dist, action_k, next_reward]

    def do_bptt_training(self):
        """Perform BPTT training with the episode history."""
        roll_ahead = self.game.roll_ahead
        look_back = self.game.look_back

        sum_loss = {"Reward": 0, "Value": 0,
                    "Policy": 0, "Latent": 0}
        for _ in range(self.mini_batch_size):
            random_epidata = random.choice(self.epidata_array)
            max_index = len(random_epidata) - roll_ahead - 1
            if max_index <= look_back:
                continue

            k = random.randint(look_back, max_index)

            states = [random_epidata[i][0]
                      for i in range(k - look_back, k + 1)]
            actions = [random_epidata[i][3]
                       for i in range(k + 1, k + roll_ahead + 1)]
            policies = [random_epidata[i][2]
                        for i in range(k, k + roll_ahead + 1)]
            values = [sum([self.discount_factor ** i for i in range(len(random_epidata) - i)])
                      for i in range(k, k + roll_ahead + 1)]
            rewards = [random_epidata[i][4]
                       for i in range(k + 1, k + roll_ahead + 1)]
            
            self.mcts.log('states=' + str(states), force=True)
            self.mcts.log('actions=' + str(actions), force=True)
            self.mcts.log('policies=' + str(policies), force=True)
            self.mcts.log('values=' + str(values), force=True)
            self.mcts.log('rewards=' + str(rewards), force=True)
            
            # TODO: Fjern MCTS fra dette callet, gjør det bare for tilgang til logger.
            loss = self.nnm.bptt(states, actions, policies, values, rewards, self.mcts)
            for loss_key, loss_value in sum_loss.items():
                sum_loss[loss_key] = loss_value + \
                    to_float(loss[loss_key]/self.mini_batch_size)
            
        # print(sum_loss)
        # self.mcts.log(sum_loss, force=True)
        for network in [self.nnm.dynamics, self.nnm.prediction, self.nnm.representation]:
            network.learning_rate *= 1





if __name__ == "__main__":
    system = System(CONFIG)
    score = system.train()


