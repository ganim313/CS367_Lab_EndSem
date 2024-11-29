import numpy as np

# Grid World dimensions and rewards
rows, cols = 3, 4
terminal_states = {(0, 3): 1, (1, 3): -1}
obstacle = (1, 1)
gamma = 0.9  # Discount factor
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# Transition probabilities
transition_probs = {
    "intended": 0.8,
    "left": 0.1,
    "right": 0.1
}

# Define the reward grid
def create_reward_grid(r_s):
    grid = np.full((rows, cols), r_s)
    for state, reward in terminal_states.items():
        grid[state] = reward
    grid[obstacle] = 0
    return grid

# Check if a position is valid
def is_valid(x, y):
    return 0 <= x < rows and 0 <= y < cols and (x, y) != obstacle

# Get next state based on action
def get_next_state(x, y, action):
    if action == "UP" and is_valid(x - 1, y):
        return x - 1, y
    if action == "DOWN" and is_valid(x + 1, y):
        return x + 1, y
    if action == "LEFT" and is_valid(x, y - 1):
        return x, y - 1
    if action == "RIGHT" and is_valid(x, y + 1):
        return x, y + 1
    return x, y  # Bump into a wall

# Perform value iteration
def value_iteration(reward_grid, threshold=1e-4):
    value_grid = np.zeros((rows, cols))
    while True:
        delta = 0
        new_value_grid = np.copy(value_grid)
        for x in range(rows):
            for y in range(cols):
                if (x, y) in terminal_states or (x, y) == obstacle:
                    continue

                # Calculate the value for each action
                action_values = []
                for action in actions:
                    value = 0
                    # Intended direction
                    nx, ny = get_next_state(x, y, action)
                    value += transition_probs["intended"] * (reward_grid[nx, ny] + gamma * value_grid[nx, ny])
                    # Left of intended direction
                    left_action = actions[(actions.index(action) - 1) % 4]
                    nx, ny = get_next_state(x, y, left_action)
                    value += transition_probs["left"] * (reward_grid[nx, ny] + gamma * value_grid[nx, ny])
                    # Right of intended direction
                    right_action = actions[(actions.index(action) + 1) % 4]
                    nx, ny = get_next_state(x, y, right_action)
                    value += transition_probs["right"] * (reward_grid[nx, ny] + gamma * value_grid[nx, ny])
                    action_values.append(value)

                # Update the value for the state
                new_value_grid[x, y] = max(action_values)
                delta = max(delta, abs(new_value_grid[x, y] - value_grid[x, y]))

        value_grid = new_value_grid
        if delta < threshold:
            break
    return value_grid

# Main function to compute values for different rewards
rewards = [-2, 0.1, 0.02, 1]
for r_s in rewards:
    print(f"\nValue function for r(s) = {r_s}:")
    reward_grid = create_reward_grid(r_s)
    value_grid = value_iteration(reward_grid)
    print(np.round(value_grid, 2))