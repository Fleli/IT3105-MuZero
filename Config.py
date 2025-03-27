CONFIG = {
    "num_episodes": 100000,
    "num_episode_steps": 500,
    "num_searches": 50,
    "max_depth": 5,
    "training_interval": 10,
    "minibatch_size": 64,
    "discount_factor": 0.95,
    "game": "gym",
    "gym": {
        "env_name": "CartPole-v1",
        "w": 5,
        "q": 0
    },
    "dynamics_nn": {
        "input_dim": 4+1 ,
        "layers": [128]*3,
        "output_dim": 4 + 1,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.0001
    },

    "prediction_nn": {
        "input_dim":  4+1,
        "layers": [128]*3,
        "output_dim": 3,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.0001
    },


    "representation_nn": {
        "input_dim": 4,
        "layers": [128]*3,
        "output_dim": 4,
        "activation_function": "relu",
        "k": 1,
        "learning_rate": 0.0001
    }
}
