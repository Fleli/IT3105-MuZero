
from MCTS.MCTSTypes import *
import jax.numpy as jnp

def dynamics_network_input(state, action):
    action_array = jnp.array([action])
    return jnp.concatenate([action_array, state])
