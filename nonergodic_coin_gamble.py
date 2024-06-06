import numpy as np
import matplotlib.pyplot as plt



# Simulation parameters for the non-ergodic system
n_simulations = 1000
n_rounds = 50
initial_wealth = 1
win_prob = 0.5

# Outcomes: heads (80% growth) and tails (50% loss)
growth_factor = 1.8
loss_factor = 0.5

# Simulate the non-ergodic outcomes
wealth_paths = np.zeros((n_simulations, n_rounds + 1))
wealth_paths[:, 0] = initial_wealth

for t in range(1, n_rounds + 1):
    outcomes = np.random.rand(n_simulations) < win_prob
    wealth_paths[outcomes, t] = wealth_paths[outcomes, t-1] * growth_factor
    wealth_paths[~outcomes, t] = wealth_paths[~outcomes, t-1] * loss_factor

# Calculate median and mean wealth
median_wealth = np.median(wealth_paths, axis=0)
mean_wealth = np.mean(wealth_paths, axis=0)

# Plot the results
plt.figure(figsize=(10, 6))
for path in wealth_paths:
    plt.plot(path, color='grey', alpha=0.1)

plt.plot(mean_wealth, linewidth=3, color='blue', label='Mean Wealth')
plt.plot(median_wealth, linewidth=3, color='red', label='Median Wealth')
# plt.axhline(y=0, color='black', linewidth=4, label='Zero Wealth')

plt.yscale('log')
plt.xlabel('Tosses')
plt.ylabel('Wealth $x(t)$')
plt.legend()
plt.title('Simulation of coin toss with 80% gain or 50% loss')

plt.grid(True, which="both", ls="--")
plt.savefig('viz/nonergodic_coin_gamble.png')
