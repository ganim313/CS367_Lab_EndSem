import yfinance as yf
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

ticker = 'AAPL' 
data = yf.download(ticker, start='2014-01-01', end='2024-01-01')

# Display the first few rows of the data
print(data.head())
# Save the data to a CSV file
data.to_csv('apple_stock_data.csv')
# Extract the 'Adj Close' column
data = data[['Adj Close']]

# Calculate daily returns
data['Returns'] = data['Adj Close'].pct_change()

# Drop the first row (which has NaN in the 'Returns' column)
data = data.dropna()

# Display the first few rows of the processed data
print(data.head())
returns = data['Returns'].values.reshape(-1, 1)

# Fit a Gaussian HMM with 2 hidden states (you can experiment with more states later)
model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000)
model.fit(returns)

# Predict the hidden states
hidden_states = model.predict(returns)

# Add the hidden states to the data
data['Hidden State'] = hidden_states
log_likelihood = model.score(returns)
print(f"Log-Likelihood: {log_likelihood:.2f}")

# Silhouette Score
silhouette = silhouette_score(returns, hidden_states)
print(f"Silhouette Score: {silhouette:.2f}")

# Davies-Bouldin Index (lower is better)
davies_bouldin = davies_bouldin_score(returns, hidden_states)
print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

# Display the first few rows of the data with hidden states
print(data.head())
means = model.means_
covariances = model.covars_

print("Means of each hidden state:")
print(means)

print("\nCovariances (variances) of each hidden state:")
print(covariances)
# Transition matrix of the HMM
transition_matrix = model.transmat_

print("Transition Matrix:")
print(transition_matrix)
plt.figure(figsize=(10, 6))
plt.plot(data['Adj Close'], label='Adjusted Close', color='black')

# Color-code the time periods based on the hidden states
plt.scatter(data.index, data['Adj Close'], c=data['Hidden State'], cmap='coolwarm', marker='.', label='Hidden State')

# Add labels and title
plt.title('Stock Prices with Inferred Hidden States')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()
# Predict the most likely next state based on the current state
current_state = hidden_states[-1]  # The last observed hidden state
next_state_probabilities = transition_matrix[current_state]

# Print the next state probabilities
print(f"Next state probabilities (current state {current_state}):")
print(next_state_probabilities)

# Predict the most likely next state
next_state = np.argmax(next_state_probabilities)
print(f"Most likely next state: {next_state}")
