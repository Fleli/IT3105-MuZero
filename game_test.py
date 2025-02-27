import pygame
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import random
from NeuralNetwork import NeuralNetwork

# For this example, we use a simple tanh activation.
activation_function = jnp.tanh

# Game settings
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
BALL_RADIUS = 10
PADDLE_WIDTH = 80
PADDLE_HEIGHT = 10

# Create a NeuralNetwork instance.
# Here we use input_dim=1 (ball x position), one hidden layer with 10 neurons, and output_layer=1.
nn = NeuralNetwork(
    input_dim=1,
    layers=[10],
    output_layer=1,
    activation_function=activation_function,
    include_bias=True,
    max_output=WINDOW_WIDTH
)

# BPTT training settings
learning_rate = 0.01
k = 20  # BPTT unrolling length

# Pre-train for a couple thousand rounds before showing the game.
num_pretrain_rounds = 20000
inputs_sequence = []
targets_sequence = []
initial_state = [jnp.zeros((layer.n_neurons,)) for layer in nn.hidden_layers]

print("Pre-training the network for {} rounds...".format(num_pretrain_rounds))
for round in tqdm(range(num_pretrain_rounds)):
    # Randomly choose a ball x position and normalize.
    ball_x = random.randint(BALL_RADIUS, WINDOW_WIDTH - BALL_RADIUS)
    norm_input = ball_x / WINDOW_WIDTH
    input_val = jnp.array([norm_input])
    target_val = jnp.array([norm_input])
    
    inputs_sequence.append(np.array(input_val))
    targets_sequence.append(np.array(target_val))
    
    # Every k rounds, run a training update via BPTT.
    if len(inputs_sequence) >= k:
        inputs_arr = jnp.array(inputs_sequence)
        targets_arr = jnp.array(targets_sequence)
        nn.BPTT(inputs_arr, targets_arr, initial_state, learning_rate, k)
        inputs_sequence = []
        targets_sequence = []
print("Pre-training finished.")

# Initialize pygame for interactive visualization.
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Neural Network Training with BPTT")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# Game variables
ball_x = random.randint(BALL_RADIUS, WINDOW_WIDTH - BALL_RADIUS)
ball_y = BALL_RADIUS
ball_speed_y = 3
paddle_y = WINDOW_HEIGHT - 50

# Reset lists for interactive training data accumulation.
inputs_sequence = []
targets_sequence = []
loss_history = []

# Main game loop (interactive)
running = True
while running:
    clock.tick(60)  # 60 fps
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Game Logic ---
    ball_y += ball_speed_y
    if ball_y > WINDOW_HEIGHT:
        ball_y = BALL_RADIUS
        ball_x = random.randint(BALL_RADIUS, WINDOW_WIDTH - BALL_RADIUS)
        # When the ball resets, if there is accumulated training data, perform a BPTT update.
        if inputs_sequence:
            inputs_arr = jnp.array(inputs_sequence)
            targets_arr = jnp.array(targets_sequence)
            nn.BPTT(inputs_arr, targets_arr, initial_state, learning_rate, k)
            # Compute loss on the last sample for visualization.
            output, _ = nn.forward(inputs_arr[-1], initial_state)
            loss = nn.loss_function(output, targets_arr[-1])
            loss_history.append(float(loss))
            if len(loss_history) > 100:
                loss_history.pop(0)
            inputs_sequence = []
            targets_sequence = []
    
    # Normalize input (ball's x position) and use as target as well.
    norm_input = ball_x / WINDOW_WIDTH
    input_val = jnp.array([norm_input])
    target_val = jnp.array([norm_input])
    inputs_sequence.append(np.array(input_val))
    targets_sequence.append(np.array(target_val))
    
    # Predict paddle position.
    init_state = [jnp.zeros((layer.n_neurons,)) for layer in nn.hidden_layers]
    pred_output, _ = nn.forward(input_val, init_state)
    paddle_x = int(float(pred_output[0]) * WINDOW_WIDTH)
    paddle_x = np.clip(paddle_x, 0, WINDOW_WIDTH - PADDLE_WIDTH)
    
    # --- Drawing ---
    screen.fill((30, 30, 30))
    pygame.draw.circle(screen, (255, 100, 100), (ball_x, int(ball_y)), BALL_RADIUS)
    pygame.draw.rect(screen, (100, 255, 100), (paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    
    # Draw a simple loss graph.
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
    
    info_text = font.render("NN paddle follows the ball. BPTT training in real time!", True, (200, 200, 200))
    screen.blit(info_text, (10, WINDOW_HEIGHT - 30))
    pygame.display.flip()

pygame.quit()
