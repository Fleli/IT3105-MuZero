CONFIG = {
    "num_episodes": 1000,
    "num_episode_steps": 100,
    "num_searches": 10,
    "max_depth": 10,
    "training_interval": 10,
    "minibatch_size": 32,

    # Configuration for game
    "game": "flappy_bird",
    # Copilot example of flappy_bird configuration
    "flappy_bird": {
        "pipe_gap": 100,
        "pipe_width": 52,
        "pipe_height": 320,
        "pipe_speed": 5,
        "bird_width": 34,
        "bird_height": 24,
        "bird_speed": 5,
        "gravity": 1,
        "jump_speed": 10,
        "screen_width": 288,
        "screen_height": 512
    },
    # Copilot example of pong configuration
    "pong": {
        "paddle_width": 10,
        "paddle_height": 50,
        "paddle_speed": 5,
        "ball_width": 10,
        "ball_height": 10,
        "ball_speed": 5,
        "screen_width": 400,
        "screen_height": 400
    },
}