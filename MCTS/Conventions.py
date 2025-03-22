
from MCTS.MCTSTypes import *

import jax
import jax.numpy as jnp

# State, Action -> Dynamics Network Input
def dynamics_network_input(state: AbstractState, action: Action, verbose=False) -> jax.Array:
    
    if verbose:
        print(f"\n{dynamics_network_input.__name__}\nstate={state}\naction={action}")
        print(type(state), state[0])
        print(state[1])
    
    inp = jnp.insert(state, 0, action)
    
    return inp

# Dynamics Network Output -> Reward, Abstract State
def dynamics_network_output(nn_output: jax.Array) -> tuple[float, jax.Array]:
    return nn_output[0], nn_output[1:]

# Prediction Network Output -> Evaluation, [Action Probabilities]
def prediction_network_output(nn_output: jax.Array) -> tuple[float, jax.Array]:
    return nn_output[0], nn_output[1:]
