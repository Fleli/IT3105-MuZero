
from MCTS.MCTSTypes import *

import jax
import jax.numpy as jnp
import jax.random as jrandom

from NeuralNetwork import *

def dynamics(state: AbstractState, action: Action, network: NeuralNetwork, params=None) -> tuple[float, AbstractState]:
    """
    Run dynamics network: (state, action) -> (reward, next_state)
    """
    inp = state.copy()
    network_input = jnp.insert(inp, 0, action)
    raw_output = network.forward(network_input, params)
    return raw_output[0], raw_output[1:]

def prediction(state: AbstractState, network: NeuralNetwork) -> tuple[float, jax.Array]:
    raw_output = network.forward(state)
    evaluation = raw_output[0]
    probabilities = jax.nn.softmax(raw_output[1:])
    return evaluation, probabilities

def weighted_choice(probabilities, options, key):
    prob_sum = sum(probabilities)
    assert prob_sum >= 0, 'Should be softmaxed first.'
    key, subkey = jrandom.split(key)
    random_float = jrandom.uniform(subkey, minval=0, maxval=prob_sum)
    cumulative = 0
    i = -1
    while cumulative <= random_float:
        i += 1
        cumulative += probabilities[i]
    return key, options[i]
