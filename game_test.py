import pygame
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import random
from NeuralNetwork import NeuralNetwork
from MCTS.MCTS import MCTS  # assuming your MCTS class is in MCTS.py

# For this example, we use a simple tanh activation.
activation_function = jnp.tanh

# Game settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BALL_RADIUS = 10
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 80

# Maximum movement speed for the paddles (pixels per frame)
MAX_SPEED = 4

# --- Define a simple game class for MCTS ---
# Here, the action space represents possible normalized vertical positions
# for the right paddle (values between 0 and 1).
class SimpleGame:
    def action_space(self):
        return [i / 100 for i in range(0, 100)]  # 0.0 to 0.99

# --- Create Neural Network Instances ---
# The networks map an abstract state (here, the normalized ball y position)
# to an action (the paddle’s vertical position).
representation_network = NeuralNetwork(
    input_dim=1,           # Concrete state is just the normalized ball y.
    layers=[10, 10, 10],
    output_layer=1,
    activation_function=activation_function,
    include_bias=True,
    max_output=WINDOW_HEIGHT - PADDLE_HEIGHT
)

dynamics_network = NeuralNetwork(
    input_dim=2,           # e.g. concatenation of abstract state and action.
    layers=[10, 10, 10],
    output_layer=1,
    activation_function=activation_function,
    include_bias=True,
    max_output=WINDOW_HEIGHT - PADDLE_HEIGHT
)

prediction_network = NeuralNetwork(
    input_dim=1,
    layers=[10, 10, 10],
    output_layer=1,
    activation_function=activation_function,
    include_bias=True,
    max_output=1  # Evaluation normalized to [0, 1].
)

# --- Instantiate MCTS ---
game = SimpleGame()
mcts = MCTS(game, dynamics_network, prediction_network, representation_network)

# --- (Optional) Pre-training of Networks ---
# For the representation network, we pretrain it with the normalized ball y positions.
learning_rate = 0.1
k = 20  # BPTT unrolling length
num_pretrain_rounds = 20000
inputs_sequence = []
targets_sequence = []
initial_state = [jnp.zeros((layer.n_neurons,)) for layer in representation_network.hidden_layers]

print("Pre-training the representation network for {} rounds...".format(num_pretrain_rounds))
for round in tqdm(range(num_pretrain_rounds)):
    ball_y = random.randint(BALL_RADIUS, WINDOW_HEIGHT - BALL_RADIUS)
    norm_input = ball_y / WINDOW_HEIGHT
    input_val = jnp.array([norm_input])
    target_val = jnp.array([norm_input])
    
    inputs_sequence.append(np.array(input_val))
    targets_sequence.append(np.array(target_val))
    
    if len(inputs_sequence) >= k:
        inputs_arr = jnp.array(inputs_sequence)
        targets_arr = jnp.array(targets_sequence)
        representation_network.BPTT(inputs_arr, targets_arr, initial_state, learning_rate, k)
        inputs_sequence = []
        targets_sequence = []
print("Pre-training finished.")

# --- Initialize pygame for interactive visualization ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("MCTS-Controlled Pong")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# Game variables
# Ball starting at center with random direction.
ball_x = WINDOW_WIDTH // 2
ball_y = WINDOW_HEIGHT // 2
ball_speed_x = random.choice([-4, 4])
ball_speed_y = random.choice([-3, 3])

# Paddles
left_paddle_x = 10
left_paddle_y = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2

right_paddle_x = WINDOW_WIDTH - (PADDLE_WIDTH + 10)
right_paddle_y = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2

# Scores
left_score = 0
right_score = 0

# For visualization of loss (if desired)
loss_history = []

# Main game loop
running = True
while running:
    clock.tick(60)  # 60 fps
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Game Logic ---
    # Update ball position.
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Bounce off top and bottom.
    if ball_y - BALL_RADIUS <= 0 or ball_y + BALL_RADIUS >= WINDOW_HEIGHT:
        ball_speed_y = -ball_speed_y

    # Check collisions with paddles.
    # Left paddle collision.
    if ball_x - BALL_RADIUS <= left_paddle_x + PADDLE_WIDTH:
        if left_paddle_y <= ball_y <= left_paddle_y + PADDLE_HEIGHT:
            ball_speed_x = -ball_speed_x
    # Right paddle collision.
    if ball_x + BALL_RADIUS >= right_paddle_x:
        if right_paddle_y <= ball_y <= right_paddle_y + PADDLE_HEIGHT:
            ball_speed_x = -ball_speed_x

    # Scoring: ball passed the paddles.
    if ball_x < 0:
        right_score += 1
        ball_x = WINDOW_WIDTH // 2
        ball_y = WINDOW_HEIGHT // 2
        ball_speed_x = 4
        ball_speed_y = random.choice([-3, 3])
    elif ball_x > WINDOW_WIDTH:
        left_score += 1
        ball_x = WINDOW_WIDTH // 2
        ball_y = WINDOW_HEIGHT // 2
        ball_speed_x = -4
        ball_speed_y = random.choice([-3, 3])

    # --- Paddle Controls ---
    # Left paddle: simple AI that tracks the ball with limited speed.
    left_target = ball_y - PADDLE_HEIGHT // 2
    if left_paddle_y < left_target:
        left_paddle_y += min(MAX_SPEED, left_target - left_paddle_y)
    elif left_paddle_y > left_target:
        left_paddle_y -= min(MAX_SPEED, left_paddle_y - left_target)
    # Ensure left paddle remains in bounds.
    left_paddle_y = np.clip(left_paddle_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)

    # Right paddle: use MCTS to choose the paddle's vertical target.
    concrete_state = jnp.array([ball_y / WINDOW_HEIGHT])
    chosen_action = mcts.search(2, [concrete_state])
    # Convert the normalized action to an absolute target position.
    target_y = int(float(chosen_action) * (WINDOW_HEIGHT - PADDLE_HEIGHT))
    
    # Gradually update right paddle position with a maximum speed.
    if right_paddle_y < target_y:
        right_paddle_y += min(MAX_SPEED, target_y - right_paddle_y)
    elif right_paddle_y > target_y:
        right_paddle_y -= min(MAX_SPEED, right_paddle_y - target_y)
    # Ensure right paddle remains in bounds.
    right_paddle_y = np.clip(right_paddle_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)

    # --- Drawing ---
    screen.fill((30, 30, 30))
    
    # Draw the ball.
    pygame.draw.circle(screen, (255, 100, 100), (ball_x, ball_y), BALL_RADIUS)
    
    # Draw the paddles.
    pygame.draw.rect(screen, (100, 255, 100), (left_paddle_x, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, (100, 255, 100), (right_paddle_x, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    
    # Draw scores.
    score_text = font.render(f"{left_score}   {right_score}", True, (255, 255, 255))
    screen.blit(score_text, (WINDOW_WIDTH // 2 - score_text.get_width() // 2, 20))
    
    # (Optional) Draw a simple loss graph if you're tracking online training loss.
    if loss_history:
        max_loss = max(loss_history)
        min_loss = min(loss_history)
        graph_height = 50
        graph_width = 200
        graph_x = WINDOW_WIDTH - graph_width - 10
        graph_y = 10
        pygame.draw.rect(screen, (50, 50, 50), (graph_x, graph_y, graph_width, graph_height))
        normalized_losses = [(l - min_loss) / (max_loss - min_loss + 1e-6) for l in loss_history]
        points = []
        for i, l in enumerate(normalized_losses):
            x = graph_x + int(i / len(normalized_losses) * graph_width)
            y = graph_y + graph_height - int(l * graph_height)
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(screen, (255, 255, 255), False, points, 2)
        loss_text = font.render(f"Loss: {loss_history[-1]:.4f}", True, (255, 255, 255))
        screen.blit(loss_text, (graph_x, graph_y + graph_height + 5))
    
    info_text = font.render("Right Paddle: MCTS | Left Paddle: Simple AI", True, (200, 200, 200))
    screen.blit(info_text, (10, WINDOW_HEIGHT - 30))
    
    pygame.display.flip()

pygame.quit()
