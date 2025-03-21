import jax
import jax.numpy as jnp
import jax.random as random
from NeuralNetwork.NeuralNetworkLayer import NeuralNetworkLayer

key = random.PRNGKey(0)

class NeuralNetwork:
    hidden_layers: list[NeuralNetworkLayer]
    include_bias: bool
    output_layer: NeuralNetworkLayer
    input_layer: NeuralNetworkLayer

    def __init__(self, input_dim: int, layers: list[int], output_dim: int, activation_function, include_bias: bool):
        self.input_layer = NeuralNetworkLayer(
            n_neurons=layers[0],
            activation_function=activation_function,
            params={
                "weights": random.normal(key, (layers[0], input_dim)) * 0.01,
                "hidden_weights": jnp.zeros((layers[0],)),  # no recurrent connection needed; or adjust as needed.
                "bias": random.normal(key, (layers[0],)) * 0.01 if include_bias else None
            },
            include_bias=include_bias
        )
        if len(layers) > 1:
            hidden_input_dims = layers[:-1]  
            self.hidden_layers = [
                NeuralNetworkLayer(
                    n_neurons=n,
                    activation_function=activation_function,
                    params={
                        "weights": random.normal(key, (n, in_dim)) * 0.01,
                        "hidden_weights": random.normal(key, (n, n)) * 0.01, 
                        "bias": random.normal(key, (n,)) * 0.01 if include_bias else None
                    },
                    include_bias=include_bias
                )
                for n, in_dim in zip(layers[1:], hidden_input_dims[1:])
            ]
        else:
            self.hidden_layers = []

        self.include_bias = include_bias
        self.reset_hidden_state()

        self.output_layer = NeuralNetworkLayer(
            n_neurons=output_dim,
            activation_function=activation_function,
            params={
                "weights": random.normal(key, (output_dim, layers[-1])) * 0.01,
                "hidden_weights": jnp.zeros((output_dim,)),  # typically no recurrent connection in output.
                "bias": random.normal(key, (output_dim,)) * 0.01 if include_bias else None
            },
            include_bias=include_bias
        )
        self.layer_parameters = [self.input_layer.parameters] + \
                                [layer.parameters for layer in self.hidden_layers] + \
                                [self.output_layer.parameters]

    def loss_function(self, activation, targets):
        return 0.5 * jnp.sum((activation - targets) ** 2)

    def forward(self, input, state_list, params=None):
        if params is None:
            params = self.layer_parameters
        current_input = self.input_layer.compute_output(input, None, params[0])
        hidden_states = []

        for i, (layer, state) in enumerate(zip(self.hidden_layers, state_list)):
            new_state = layer.compute_output(current_input, state, params[i + 1])
            hidden_states.append(new_state)
            current_input = new_state

        output = self.output_layer.compute_output(current_input, None, params[-1])
        return output, hidden_states

    def backward(self, stored_state, stored_input, target):
        def loss_fn(layer_params):
            output, _ = self.forward(stored_input, stored_state, layer_params)
            return self.loss_function(output, target)

        grad_layer_parameters = jax.grad(loss_fn)(self.layer_parameters)
        return grad_layer_parameters

    def BPTT(self, inputs, targets, initial_state, learning_rate, k):
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

            if ((t + 1) % k == 0) or (t == T - 1):
                block_length = len(stored_inputs)
                block_offset = t - block_length + 1

                grad_layer_parameters = jax.tree_map(jnp.zeros_like, self.layer_parameters)

                
                for j in range(block_length):
                    target_idx = block_offset + j
                    grad_param_j = self.backward(stored_states[j], stored_inputs[j], targets[target_idx])
                    grad_layer_parameters = jax.tree_map(lambda g1, g2: g1 + g2,
                                                         grad_layer_parameters, grad_param_j)
                
                self.layer_parameters = jax.tree_map(lambda param, grad: param - learning_rate * grad,
                                                     self.layer_parameters, grad_layer_parameters)
                
                self.input_layer.parameters = self.layer_parameters[0]
                for i, layer in enumerate(self.hidden_layers):
                    layer.parameters = self.layer_parameters[i + 1]
                self.output_layer.parameters = self.layer_parameters[-1]

                stored_states = [state]
                stored_inputs = []
                stored_targets = []
        
    def predict(self, input):
        output, state = self.forward(input, self.hidden_states)
        self.hidden_states = state
        return output, state

    def reset_hidden_state(self):
        self.hidden_states = [jnp.zeros((layer.n_neurons,)) for layer in self.hidden_layers]
