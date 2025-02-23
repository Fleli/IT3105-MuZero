
from MCTSTypes import *

def dynamics_network_input( state: AbstractState, action: Action ) -> list[float]:
    return [action] + state
