CONFIG = {
    "num_episodes": 16*50,
    "num_episode_steps": 500,
    "num_searches": 10,
    "max_depth": 5,
    "training_interval": 16,
    "minibatch_size": 16,
    "discount_factor": 1,
    "exploration": 1.25,
    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,
        "q": 0
    },
    "dynamics_nn": {
        "input_dim": 4+1 ,
        "layers": [128]*4,
        "output_dim": 4 + 1,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.001
    },

    "prediction_nn": {
        "input_dim":  4+1,
        "layers": [128]*4,
        "output_dim": 3,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.001
    },


    "representation_nn": {
        "input_dim": 4,
        "layers": [128]*4,
        "output_dim": 4,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.001
    }
}
