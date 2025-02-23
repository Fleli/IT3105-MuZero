import jax
import jax.numpy as jnp
from NeuralNetwork.NeuralNetworkLayer import NeuralNetworkLayer


class NeuralNetwork:
    hidden_layers: list[int]
    include_bias: bool
    final_hidden_layer: NeuralNetworkLayer = None
    output_layer: NeuralNetworkLayer = None

    def __init__(self, layers: list[int], output_layer: int, activation_function, include_bias: bool, max_output: float):
        self.hidden_layers = layers
        self.max_output = max_output
        self.include_bias = include_bias
        for n_neurons_in_layer in layers:
            self.final_hidden_layer = NeuralNetworkLayer(
                n_neurons_in_layer, self.final_hidden_layer, activation_function, include_bias)
        self.output_layer = NeuralNetworkLayer(output_layer, None,
                           activation_function, include_bias)

    def loss_function(self, activation, targets):
        return 0.5 * jnp.sum((activation - targets) ** 2)

    def forward(self, input, state):
        new_hidden_state = self.final_hidden_layer.compute_output(jnp.concatenate([input, state], axis=0))
        output = self.output_layer.compute_output(new_hidden_state)
        return output, new_hidden_state 

    def backward(self, grad_output, grad_state_next, stored_state, input):
        def forward_fn(params, state):
            original_params = self.parameters
            self.parameters = params
            output, new_state = self.forward(input, state)
            self.parameters = original_params
            return output, new_state

        (output, new_state), vjp_fn = jax.vjp(
            forward_fn, self.parameters, stored_state)
        combined_grad = (grad_output, grad_state_next)
        grad_parameters, grad_state = vjp_fn(combined_grad)

        return grad_parameters, grad_state

    def BPTT(self, inputs, targets, initial_state, model, learning_rate, k):
        T = len(inputs)  # Number of time steps in series

        state = initial_state
        stored_states = [state]
        stored_activations = []
        stored_losses = []

        grad_loss_fn = jax.grad(self.loss_function, argnums=0)
        for t in range(0, T-1):
            output, state = model.forward(inputs[t], state)
            activation = output
            loss = self.loss_function(activation, targets[t])

            stored_activations.append(activation)
            stored_losses.append(loss)
            stored_states.append(state)

            if ((t + 1) % k == 0) or (t == T-1):
                grad_state = 0
                grad_parameters = jax.tree_map(
                    jnp.zeros_like, self.parameters)

                for j in range(t, max(0, t - k + 1), -1):
                    grad_output = grad_loss_fn(stored_activations[j], targets[j])
                    grad_param_j, grad_state = self.backward(
                        grad_output, grad_state, stored_states[j], inputs[j])  # Computes gradient at the specified timestep
                    grad_parameters = grad_parameters + grad_param_j

                self.parameters = self.parameters - learning_rate * grad_parameters

                # state = detach(state) #Makes new state not needed in jax since jax arrays are immutable
                stored_states = [state]
                stored_activations = []
                stored_losses = []
