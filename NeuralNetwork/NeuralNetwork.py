import jax
import jax.numpy as jnp
import jax.random as random
from NeuralNetwork.NeuralNetworkLayer import NeuralNetworkLayer

key = random.PRNGKey(0)

class NeuralNetwork:
    hidden_layers: list[NeuralNetworkLayer]
    include_bias: bool
    output_layer: NeuralNetworkLayer

    def __init__(self, input_dim: int, layers: list[int], output_layer: int, activation_function, include_bias: bool, max_output: float):
        layer_input_dims = [input_dim] + layers[:-1]
        self.hidden_layers = [
            NeuralNetworkLayer(
                n_neurons=n,
                activation_function=activation_function,
                params={
                    "weights": random.normal(key, (n, in_dim)) * 0.1,
                    "hidden_weights": random.normal(key, (n, n)) * 0.1,
                    "bias": random.normal(key, (n,)) * 0.1 if include_bias else None
                },
                include_bias=include_bias
            )
            for n, in_dim in zip(layers, layer_input_dims)
        ]
        self.max_output = max_output
        self.include_bias = include_bias
        self.reset_hidden_state()

        self.output_layer = NeuralNetworkLayer(
            n_neurons=output_layer,
            activation_function=activation_function,
            params={
                "weights": random.normal(key, (output_layer, layers[-1])) * 0.1,
                "hidden_weights": jnp.zeros((output_layer,)),
                "bias": random.normal(key, (output_layer,)) * 0.1 if include_bias else None
            },
            include_bias=include_bias
        )

        self.layer_parameters = [
            layer.parameters for layer in self.hidden_layers] + [self.output_layer.parameters]

    def loss_function(self, activation, targets):
        return 0.5 * jnp.sum((activation - targets) ** 2)

    def forward(self, input, state_list, params=None):
        if params is None:
            params = self.layer_parameters
        hidden_states = []
        current_input = input
        for i, (layer, state) in enumerate(zip(self.hidden_layers, state_list)):
            new_state = layer.compute_output(current_input, state, params[i])
            hidden_states.append(new_state)
            current_input = new_state
        output = self.output_layer.compute_output(current_input)
        return output, hidden_states

    def backward(self, grad_output, grad_state_next, stored_state, input):
        def forward_fn(layer_params, state):
            return self.forward(input, state, layer_params)
        _, vjp_fn = jax.vjp(forward_fn, self.layer_parameters, stored_state)
        combined_grad = (grad_output, grad_state_next)
        grad_layer_parameters, grad_state = vjp_fn(combined_grad)
        return grad_layer_parameters, grad_state

    def BPTT(self, inputs, targets, initial_state, learning_rate, k):
        T = len(inputs)  # total time steps

        state = initial_state
        stored_states = [state]
        stored_activations = []
        stored_inputs = []   # store the inputs for the current block
        grad_loss_fn = jax.grad(self.loss_function, argnums=0)

        for t in range(T):
            output, state = self.forward(inputs[t], state)
            activation = output
            loss = self.loss_function(activation, targets[t])
            stored_activations.append(activation)
            stored_states.append(state)
            stored_inputs.append(inputs[t])

            if ((t + 1) % k == 0) or (t == T - 1):
                # Compute the offset for the current block.
                block_length = len(stored_activations)
                block_offset = t - block_length + 1

                grad_state = jax.tree_map(jnp.zeros_like, initial_state)
                grad_layer_parameters = jax.tree_map(
                    jnp.zeros_like, self.layer_parameters)

                for j in range(block_length - 1, -1, -1):
                    target_idx = block_offset + j
                    grad_output = grad_loss_fn(
                        stored_activations[j], targets[target_idx])
                    grad_param_j, grad_state = self.backward(
                        grad_output, grad_state, stored_states[j], stored_inputs[j])
                    grad_layer_parameters = jax.tree_map(lambda g1, g2: g1 + g2,
                                                         grad_layer_parameters, grad_param_j)

                self.layer_parameters = jax.tree_map(lambda param, grad: param - learning_rate * grad,
                                                     self.layer_parameters, grad_layer_parameters)
                for i, layer in enumerate(self.hidden_layers):
                    layer.parameters = self.layer_parameters[i]
                self.output_layer.parameters = self.layer_parameters[-1]

                stored_states = [state]
                stored_activations = []
                stored_inputs = []

    def predict(self, input):
        output, state = self.forward(input, self.hidden_states)
        self.hidden_states = state
        return output

    def reset_hidden_state(self):
        self.hidden_states = [jnp.zeros((layer.n_neurons, ))
                              for layer in self.hidden_layers]
