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

    def __init__(self, config):
        input_dim = config['input_dim']
        layers = config['layers']
        output_dim = config['output_dim']
        activation_function = config['activation_function']
        include_bias = config['include_bias']
        self.k = config['k']
        self.learning_rate = config['learning_rate']
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

    def backward(self, stored_state, stored_input, target):         # Denne blir aldri called. Skal vel bruke variant i BPTT.
        def loss_fn(layer_params):
            output, _ = self.forward(stored_input, stored_state, layer_params)
            return self.loss_function(output, target)

        grad_layer_parameters = jax.grad(loss_fn)(self.layer_parameters)
        return grad_layer_parameters

    def predict(self, input):
        output, state = self.forward(input, self.hidden_states)
        self.hidden_states = state
        return output, state

    def reset_hidden_state(self):
        self.hidden_states = [jnp.zeros((layer.n_neurons,)) for layer in self.hidden_layers]
