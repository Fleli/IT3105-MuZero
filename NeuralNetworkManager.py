
from NeuralNetwork import *
import jax
import jax.numpy as jnp
from MCTS.Conventions import dynamics_network_output, prediction_network_output, dynamics_network_input

def prediction_loss(raw_value, raw_policy_logits, target_value, target_policy):
    value_loss = jnp.square(raw_value - target_value)
    
    target_policy = target_policy / jnp.sum(target_policy)
    log_probs = jax.nn.log_softmax(raw_policy_logits)
    policy_loss = -jnp.sum(target_policy * log_probs)
    return value_loss, policy_loss

class NeuralNetworkManager():
    dynamics: NeuralNetwork
    prediction: NeuralNetwork
    representation: NeuralNetwork
    
    
    def __init__(self, CONFIG):
        self.dynamics = NeuralNetwork(CONFIG["dynamics_nn"])
        self.prediction = NeuralNetwork(CONFIG["prediction_nn"])
        self.representation = NeuralNetwork(CONFIG["representation_nn"])
        self.look_back = CONFIG['gym']['q']
    
    def gather_states(self, states, state, k):

        concrete_state = []

        for state_index in range(k - self.look_back, k):
            if state_index <= 0:
                concrete_state.append(jnp.zeros_like(state))
            else:
                concrete_state.append(states[state_index])
        return jnp.array(concrete_state + [state]).flatten()
    
    def bptt(self, states, actions, policies, values, rewards):
        T = len(actions) 
        def composite_loss(comp_params, split_loss=False):
            dynamics_params, prediction_params, representation_params = comp_params
            total_loss = 0.0
            concrete_state = self.gather_states([], states[0], 0)
            latent = self.representation.forward(concrete_state, representation_params)

            pred_out = self.prediction.forward(jnp.concatenate([jnp.array([rewards[0]]),latent]), prediction_params)
            raw_value = pred_out[0]
            raw_policy_logits = pred_out[1:]
            
            policy = policies[0]
            counts = jnp.array([policy[0], policy[1]])
            target_policy = counts / jnp.sum(counts)
            target_value = jnp.array([values[0]])
            
            total_value_loss, total_policy_loss = prediction_loss(raw_value, raw_policy_logits, target_value, target_policy)
            total_latent_loss = 0
            total_reward_loss = 0

            for t in range(T):
                dynamics_input = dynamics_network_input(latent, actions[t])
                reward, pred_latent = dynamics_network_output(self.dynamics.forward(dynamics_input, dynamics_params))
                reward_target = jnp.array([rewards[t]])
                reward_loss = self.dynamics.loss_function(reward, reward_target)
                latent_loss = 0.0
                if t + 1 < len(states):
                    concrete_state = self.gather_states(states, states[t+1], t+1)
                    latent = self.representation.forward(concrete_state, representation_params)
                    latent_loss = self.dynamics.loss_function(pred_latent, latent)


                pred_in = jnp.concatenate([jnp.array([reward]), latent])
                pred_out = self.prediction.forward(pred_in, prediction_params)
                raw_value = pred_out[0]
                raw_policy_logits = pred_out[1:]
                
                policy = policies[t + 1]
                counts = jnp.array([policy[0], policy[1]])
                target_policy = counts / jnp.sum(counts)
                target_value = jnp.array([values[t + 1]])
                
                value_loss, policy_loss = prediction_loss(raw_value, raw_policy_logits, target_value, target_policy)

                total_value_loss += value_loss
                total_policy_loss += policy_loss
                total_latent_loss += latent_loss
                total_reward_loss += reward_loss

            total_loss = total_reward_loss + total_value_loss + total_policy_loss + total_latent_loss
            if split_loss:
                return {"Reward":total_reward_loss/(T+1), "Value": total_value_loss/(T+1), 
                        "Policy": total_policy_loss/(T+1), "Latent": total_latent_loss/(T+1)}

            else:
                return jnp.sum(total_loss)


        
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
        loss_value = composite_loss(comp_params, split_loss=True)
        return loss_value