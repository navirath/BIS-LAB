import numpy as np
import matplotlib.pyplot as plt

# Step 2: Initialize Parameters
N = 100
beta = 0.3
gamma = 0.1
iterations = 100

# 0 = Healthy, 1 = Infected, 2 = Recovered
grid = np.zeros((N, N), dtype=int)

# Step 3: Initialize Population
initial_infected = 10
infected_x = np.random.randint(0, N, initial_infected)
infected_y = np.random.randint(0, N, initial_infected)
grid[infected_x, infected_y] = 1

# Helper: count infected neighbors
def count_infected_neighbors(grid, x, y):
    neighbors = grid[max(0, x-1):min(N, x+2), max(0, y-1):min(N, y+2)]
    infected = np.sum(neighbors == 1)
    if grid[x, y] == 1:
        infected -= 1
    return infected

# Data lists for analysis
infected_counts, recovered_counts, healthy_counts = [], [], []

# Step 4–6: Evaluate, Update, Iterate
for t in range(iterations):
    new_grid = grid.copy()

    for i in range(N):
        for j in range(N):
            state = grid[i, j]
            if state == 0:  # Healthy
                infected_neighbors = count_infected_neighbors(grid, i, j)
                infection_prob = 1 - (1 - beta) ** infected_neighbors
                if np.random.rand() < infection_prob:
                    new_grid[i, j] = 1  # becomes infected
            elif state == 1:  # Infected
                if np.random.rand() < gamma:
                    new_grid[i, j] = 2  # recovers

    grid = new_grid.copy()

    # Track data
    infected_counts.append(np.sum(grid == 1))
    recovered_counts.append(np.sum(grid == 2))
    healthy_counts.append(np.sum(grid == 0))

    # Live visualization
    plt.imshow(grid, cmap='viridis')
    plt.title(f"Iteration {t+1}")
    plt.axis('off')
    plt.pause(0.1)

plt.show()

# Step 7: Output Best Solution
plt.figure(figsize=(8,5))
plt.plot(healthy_counts, color='green', label='Healthy')
plt.plot(infected_counts, color='red', label='Infected')
plt.plot(recovered_counts, color='blue', label='Recovered')
plt.xlabel("Iterations")
plt.ylabel("Population Count")
plt.title("Disease Spread Over Time (PCA Simulation)")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Final Results ---")
print(f"Healthy:   {healthy_counts[-1]}")
print(f"Infected:  {infected_counts[-1]}")
print(f"Recovered: {recovered_counts[-1]}")
