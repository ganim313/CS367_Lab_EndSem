import random
import matplotlib.pyplot as plt

# --- Bandit ---
class BinaryBandit(object):
    def __init__(self, probabilities):
        # Set the number of arms (actions) based on the length of probabilities
        self.N = len(probabilities)
        self.p = probabilities  # Set the given probabilities directly

    def actions(self):
        return list(range(self.N))  # Returns available actions (0, 1, ..., N-1)

    def reward(self, action):
        rand = random.random()
        if rand < self.p[action]:
            return 1  # Success
        else:
            return 0  # Failure

def eGreedy_binary(myBandit, epsilon, max_iteration):
    # Initialization 
    Q = [0] * myBandit.N  # Estimated value for each action
    count = [0] * myBandit.N  # Count of actions taken
    R = []  # Rewards collected
    R_avg = [0]  # Average reward over iterations

    # Incremental Implementation
    for iter in range(1, max_iteration + 1):
        if random.random() > epsilon:
            action = Q.index(max(Q))  # Exploit: choose the action with max estimated value
        else:
            action = random.choice(myBandit.actions())  # Explore: choose a random action
        
        r = myBandit.reward(action)  # Get the reward for the chosen action
        R.append(r)
        count[action] += 1
        Q[action] += (r - Q[action]) / count[action]  # Update Q value using incremental formula
        R_avg.append(R_avg[iter - 1] + (r - R_avg[iter - 1]) / iter)

        # At the final iteration, print results and plot
        if iter == max_iteration:
            print("Final Counts for move 1:", count[0])
            print("Final Counts for move 2:", count[1])

            actionTaken = ["1", "2"]
            # Create bar plot
            plt.bar(actionTaken, count)
            plt.title("Number of times each action taken")
            plt.xlabel("Action")
            plt.ylabel("Count")
            plt.show()

    return Q, R_avg, R

# Set the probabilities for Bandit A and Bandit B
probabilities_A = [0.1, 0.2]  # Probabilities for Bandit A
probabilities_B = [0.8, 0.9]   # Probabilities for Bandit B

# Run for Bandit A
random.seed(3)  # Set random seed for reproducibility
myBanditA = BinaryBandit(probabilities_A)  # Create an instance for Bandit A
Q_A, R_avg_A, R_A = eGreedy_binary(myBanditA, 0.1, 2000)  # Run the epsilon-greedy algorithm

# Print results for Bandit A
print("******************        RESULTS FOR BANDIT A         *************************")
print("Observed Average Reward over 2000 experiments for action 1:", Q_A[0])
print("Observed Average Reward over 2000 experiments for action 2:", Q_A[1])
print("----------------------------------------------------------------------------------")
print("Actual Reward for action 1:", myBanditA.p[0])
print("Actual Reward for action 2:", myBanditA.p[1])
print("**********************************************************************************")

# Display the images for Bandit A
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(R_avg_A)  # Plot average rewards
ax1.title.set_text("Average Reward V/s Iteration for Bandit A")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(R_A)  # Plot individual rewards
ax2.title.set_text("Reward V/s iteration for Bandit A")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")
plt.show()

# Run for Bandit B
random.seed(9)  # Set random seed for reproducibility
myBanditB = BinaryBandit(probabilities_B)  # Create an instance for Bandit B
Q_B, R_avg_B, R_B = eGreedy_binary(myBanditB, 0.1, 2000)  # Run the epsilon-greedy algorithm

# Print results for Bandit B
print("************************        RESULTS FOR BANDIT B         ******************************")
print("Observed Average Reward over 2000 experiments for action 1:", Q_B[0])
print("Observed Average Reward over 2000 experiments for action 2:", Q_B[1])
print("---------------------------------------------------------------------------------------------")
print("Actual Reward for action 1:", myBanditB.p[0])
print("Actual Reward for action 2:", myBanditB.p[1])
print("**********************************************************************************************")

# Display the images for Bandit B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(R_avg_B)  # Plot average rewards
ax1.title.set_text("Average Reward V/s Iteration for Bandit B")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(R_B)  # Plot individual rewards
ax2.title.set_text("Reward V/s iteration for Bandit B")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")
plt.show()
