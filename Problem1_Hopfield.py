import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.flatten()  # Flatten the pattern to 1D
            self.weights += np.outer(pattern, pattern)  # Outer product
        # Set the diagonal to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)

    def update(self, state):
        # Update the state of the network
        for i in range(self.size):
            input_sum = np.dot(self.weights[i], state)
            state[i] = 1 if input_sum > 0 else 0  # Activation function (binary step)
        return state

    def recall(self, input_pattern, steps=5):
        state = input_pattern.flatten()
        for _ in range(steps):
            state = self.update(state)
        return state.reshape((10, 10))  # Reshape back to 10x10

# Example usage
if __name__ == "__main__":
    size = 100  # 10x10
    hopfield_net = HopfieldNetwork(size)

    # Define some binary patterns (10x10)
    pattern1 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    pattern2 = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    # Train the Hopfield network with the defined patterns
    hopfield_net.train([pattern1, pattern2])

    # Test the recall function with a noisy version of pattern1
    noisy_input = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Recall the pattern from the noisy input
    recalled_pattern = hopfield_net.recall(noisy_input)

    print("Recalled Pattern:")
    print(recalled_pattern)  # Output the recalled pattern