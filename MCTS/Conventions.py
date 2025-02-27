
from MCTSTypes import *

# State, Action -> Dynamics Network Input
def dynamics_network_input(state: AbstractState, action: Action) -> list[float]:
    return [action] + state

# Dynamics Network Output -> Reward, Abstract State
def dynamics_network_output(nn_output: list[float]) -> tuple[float, list[float]]:
    return nn_output[0], nn_output[1:]

# Prediction Network Output -> Evaluation, [Action Probabilities]
def prediction_network_output(nn_output: list[float]) -> tuple[float, list[float]]:
    return nn_output[0], nn_output[1:]
