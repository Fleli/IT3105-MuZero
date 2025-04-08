
from MCTS.MCTSTypes import *

import jax
import jax.numpy as jnp

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
    logits = raw_output[1:]
    probabilities = jax.nn.softmax(logits)
    return evaluation, probabilities

