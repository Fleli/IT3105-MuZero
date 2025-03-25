
from NeuralNetwork import *

class NeuralNetworkManager():
    
    # The three networks
    dynamics: NeuralNetwork
    prediction: NeuralNetwork
    representation: NeuralNetwork
    
    
    # Initialization
    def __init__(self, CONFIG):
        self.dynamics = NeuralNetwork(CONFIG["dynamics_nn"])
        self.prediction = NeuralNetwork(CONFIG["prediction_nn"])
        self.representation = NeuralNetwork(CONFIG["representation_nn"])
    
    
    # Backpropagation through time (simultaneous training)
    def bptt(self, states, actions, policies, values, rewards):
        
        T = len(states)
        
        # List of ALL parameters in composite network.
        composite_parameters = self.dynamics.layer_parameters + \
                                self.prediction.layer_parameters + \
                                self.representation.layer_parameters
        
        for t in range(T):
            pass
        
        assert False, 'Implementation is incomplete.'

    
    """
    Fra NeuralNetwork:
    
    def BPTT(self, inputs, targets, initial_state):
        T = len(inputs) 

        state = initial_state
        stored_states = [state]
        stored_inputs = []   
        stored_targets = [] 

        for t in range(T):
            output, state = self.forward(inputs[t], state)
            loss = self.loss_function(output, targets[t])

            stored_states.append(state)
            stored_inputs.append(inputs[t])
            stored_targets.append(targets[t])

            if ((t + 1) % self.k == 0) or (t == T - 1):
                block_length = len(stored_inputs)
                block_offset = t - block_length + 1

                grad_layer_parameters = jax.tree_map(jnp.zeros_like, self.layer_parameters)

                
                for j in range(block_length):
                    target_idx = block_offset + j
                    grad_param_j = self.backward(stored_states[j], stored_inputs[j], targets[target_idx])
                    grad_layer_parameters = jax.tree_map(lambda g1, g2: g1 + g2,
                                                         grad_layer_parameters, grad_param_j)
                
                self.layer_parameters = jax.tree_map(lambda param, grad: param - self.learning_rate * grad,
                                                     self.layer_parameters, grad_layer_parameters)
                
                self.input_layer.parameters = self.layer_parameters[0]
                for i, layer in enumerate(self.hidden_layers):
                    layer.parameters = self.layer_parameters[i + 1]
                self.output_layer.parameters = self.layer_parameters[-1]

                stored_states = [state]
                stored_inputs = []
                stored_targets = []
        
    """