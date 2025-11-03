import matplotlib.pyplot as plt
import numpy as np

# Observed outcomes of coin flips (1 for heads, 0 for tails)
observations = np.array([1, 0, 1, 1, 0, 1, 0, 1])

# Generate initial ensemble of probabilities based on prior knowledge
# Assume a Gaussian distribution centered at 0.5 with a standard deviation of 0.1
n_ensemble = 5
ensemble = np.clip(np.random.normal(0.5, 0.1, n_ensemble), 0, 1)

# Number of iterations for the IES/ESI
n_iterations = 10

# Store ensemble evolution and realizations for visualization
ensemble_history = [ensemble.copy()]
realizations_history = []
weights_history = []
final_weighted_predictions = []

# Function to simulate coin flips given a probability p
def simulate_flips(p, n_flips):
    return np.random.binomial(1, p, n_flips)

# Iterative Ensemble Smoothing (IES)
for iteration in range(n_iterations):
    simulated_outcomes = np.array([simulate_flips(p, len(observations)) for p in ensemble])
    misfits = simulated_outcomes - observations[np.newaxis, :]

    # Store current realizations
    realizations_history.append(simulated_outcomes)

    # Calculate mean misfit for each ensemble member
    mean_misfit = np.mean(misfits, axis=1)

    # Update ensemble members based on the mean misfit
    ensemble -= 0.1 * mean_misfit  # 0.1 is a learning rate to control update size

    # Ensure probabilities remain valid (between 0 and 1)
    ensemble = np.clip(ensemble, 0, 1)

    # Store ensemble state for visualization
    ensemble_history.append(ensemble.copy())

    # Print the updated ensemble after each iteration
    print(f"Iteration {iteration + 1}: {ensemble}")

# Plot ensemble evolution for IES
plt.figure(figsize=(10, 6))
for i in range(n_ensemble):
    plt.plot([ens[i] for ens in ensemble_history], marker='o', label=f'Ensemble Member {i+1}')
plt.title('Ensemble Evolution Over Iterations (IES)')
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

# Visualize realizations for each ensemble member over iterations
plt.figure(figsize=(12, 8))
for iteration, realizations in enumerate(realizations_history):
    plt.subplot(n_iterations // 2 + 1, 2, iteration + 1)
    for i in range(n_ensemble):
        plt.plot(realizations[i], marker='o', linestyle='-', label=f'Member {i+1}' if iteration == 0 else "")
    plt.title(f'Realizations at Iteration {iteration + 1}')
    plt.xlabel('Flip Index')
    plt.ylabel('Outcome')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(1.05, 1), ncol=1)
plt.show()

# Ensemble Space Inversion (ESI)
weights = np.ones(n_ensemble) / n_ensemble  # Start with equal weights
for iteration in range(n_iterations):
    simulated_outcomes = np.array([simulate_flips(p, len(observations)) for p in ensemble])

    # Weighted average prediction
    weighted_prediction = np.dot(weights, simulated_outcomes)
    misfit = weighted_prediction - observations

    # Update weights based on the misfit (simplified step)
    weights -= 0.05 * np.dot(misfit, simulated_outcomes.T)  # Adjust weights
    weights = np.clip(weights, 0, 1)  # Ensure weights remain non-negative
    weights /= np.sum(weights)  # Normalize to sum to 1

    # Store weights history for visualization
    weights_history.append(weights.copy())
    final_weighted_predictions.append(np.dot(weights, ensemble))

    # Print the updated weights after each iteration
    print(f"ESI Iteration {iteration + 1}: {weights}")

# Plot weights evolution for ESI
plt.figure(figsize=(10, 6))
for i in range(n_ensemble):
    plt.plot([weights[i] for weights in weights_history], marker='o', label=f'Weight {i+1}')
plt.title('Weight Evolution Over Iterations (ESI)')
plt.xlabel('Iteration')
plt.ylabel('Weight')
plt.legend()
plt.grid(True)
plt.show()

# Final estimate of p from ESI
final_estimate = np.dot(weights, ensemble)
print(f"Final ESI estimate of p: {final_estimate:.3f}")

# Plot final weighted predictions
plt.figure(figsize=(10, 6))
plt.plot(final_weighted_predictions, marker='o', color='b')
plt.title('Final Weighted Predictions Over Iterations (ESI)')
plt.xlabel('Iteration')
plt.ylabel('Final Weighted Prediction of p')
plt.grid(True)
plt.show()
