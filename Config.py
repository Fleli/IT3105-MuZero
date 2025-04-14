CONFIG = {
    "num_episodes": 16*500,
    "num_episode_steps": 500,
    "num_searches": 8,
    "max_depth": 1,
    "verbose": False,
    "training_interval": 32,
    "minibatch_size": 16,
    "discount_factor": 1,
    "exploration": 1,
    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,
        "q": 5
    },
    "dynamics_nn": {
        "input_dim": 24+1 ,
        "layers": [16]*4,
        "output_dim": 24 + 1,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.001
    },

    "prediction_nn": {
        "input_dim": 24,
        "layers": [16]*4,
        "output_dim": 3,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.001
    },


    "representation_nn": {
        "input_dim": 24,
        "layers": [24]*4,
        "output_dim": 24,
        "activation_function": "sigmoid",
        "k": 1,
        "learning_rate": 0.001
    }
}
