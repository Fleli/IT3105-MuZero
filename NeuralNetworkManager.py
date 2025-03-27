
from NeuralNetwork import *
import jax
import jax.numpy as jnp
from MCTS.Conventions import dynamics_network_output, prediction_network_output, dynamics_network_input

class NeuralNetworkManager():
    dynamics: NeuralNetwork
    prediction: NeuralNetwork
    representation: NeuralNetwork
    
    
    def __init__(self, CONFIG):
        self.dynamics = NeuralNetwork(CONFIG["dynamics_nn"])
        self.prediction = NeuralNetwork(CONFIG["prediction_nn"])
        self.representation = NeuralNetwork(CONFIG["representation_nn"])
    
    
    def bptt(self, states, actions, policies, values, rewards):
        T = len(actions) 
        def composite_loss(comp_params):
            dynamics_params, prediction_params, representation_params = comp_params
            total_loss = 0.0

            latent = self.representation.forward(states[0], representation_params)

            evaluation, pred = prediction_network_output(self.prediction.forward(jnp.concatenate([jnp.array([rewards[0]]),latent]), prediction_params))
            policy_target = jnp.array([policies[0][0], policies[0][1]])
            value_target = jnp.array([values[0]])
            pred_target = jnp.concatenate([value_target, policy_target])

            total_loss += self.prediction.loss_function(jnp.concatenate([jnp.array([evaluation]), pred]), pred_target)

            for t in range(T):
                dynamics_input = dynamics_network_input(latent, actions[t])
                reward, latent = dynamics_network_output(self.dynamics.forward(dynamics_input, dynamics_params))
                reward_target = jnp.array([rewards[t]])
                reward_loss = self.dynamics.loss_function(reward, reward_target)
                latent_loss = 0.0
                if t + 1 < len(states):
                    target_latent = self.representation.forward(states[t+1], representation_params)
                    latent_loss = self.dynamics.loss_function(latent, target_latent)


                evaluation, pred = prediction_network_output(self.prediction.forward(jnp.concatenate([jnp.array([reward]),latent]), prediction_params))
                policy = policies[t + 1]
                counts = jnp.array([policy[0], policy[1]])
                policy_target = counts / jnp.sum(counts)
                value_target = jnp.array([values[t + 1]])
                pred_target = jnp.concatenate([value_target, policy_target])
                pred_loss = self.prediction.loss_function(jnp.concatenate([jnp.array([evaluation]), pred]), pred_target)

                total_loss += reward_loss + pred_loss + latent_loss

            return total_loss

        
        comp_params = (
            self.dynamics.layer_parameters,
            self.prediction.layer_parameters,
            self.representation.layer_parameters
        )

        grads = jax.grad(composite_loss)(comp_params)

        self.dynamics.layer_parameters = jax.tree_map(
            lambda p, g: p - self.dynamics.learning_rate * g,
            self.dynamics.layer_parameters, grads[0]
        )
        self.prediction.layer_parameters = jax.tree_map(
            lambda p, g: p - self.prediction.learning_rate * g,
            self.prediction.layer_parameters, grads[1]
        )
        self.representation.layer_parameters = jax.tree_map(
            lambda p, g: p - self.representation.learning_rate * g,
            self.representation.layer_parameters, grads[2]
        )
