import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        
    def train(self, patterns):
        for pattern in patterns:
            bipolar_pattern = 2 * pattern - 1  # Convert to bipolar (-1, 1)
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        np.fill_diagonal(self.weights, 0)  # No self-connection
        
    def update(self, state):
        for i in range(self.size):
            net_input = np.dot(self.weights[i], state)
            state[i] = 1 if net_input > 0 else 0  # Update state
        return state
    
    def run(self, initial_state, steps=10):
        state = initial_state.copy()
        for _ in range(steps):
            state = self.update(state)
        return state

# Eight-Rook Problem Implementation
def eight_rook_problem():
    # Define the patterns for placing 8 rooks
    patterns = []
    for i in range(8):
        pattern = np.zeros(64)  # 8x8 board flattened
        for j in range(8):
            if j == i:
                pattern[i * 8 + j] = 1  # Place rook in row i, column j
        patterns.append(pattern)
    
    # Create the Hopfield network
    hopfield_net = HopfieldNetwork(size=64)  # 8x8 grid = 64 neurons
    hopfield_net.train(patterns)
    
    # Initialize a random state
    initial_state = np.random.choice([0, 1], size=64)
    
    # Run the network
    final_state = hopfield_net.run(initial_state, steps=100)
    
    # Reshape final state to 8x8 grid
    final_board = final_state.reshape((8, 8))
    print("Final State (8-Rook):")
    print(final_board)

# Run the Eight-Rook problem
eight_rook_problem()