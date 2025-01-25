import matplotlib.pyplot as plt

# Initialize lists to store data
depths_hill_climbing = []
nodes_hill_climbing = []
depths_iterative = []
nodes_iterative = []
depths_vnd = []
nodes_vnd = []

# Read from the output file
with open('results.txt', 'r') as file:
    lines = file.readlines()
    
    for i, line in enumerate(lines):
        depth, nodes = map(int, line.strip().split(','))
        
        # Distributing the results based on the iteration
        if i % 3 == 0:  # Hill Climbing
            depths_hill_climbing.append(depth)
            nodes_hill_climbing.append(nodes)
        elif i % 3 == 1:  # Iterative Hill Climbing
            depths_iterative.append(depth)
            nodes_iterative.append(nodes)
        elif i % 3 == 2:  # Variable Neighbourhood Descent
            depths_vnd.append(depth)
            nodes_vnd.append(nodes)

# Plotting the results
plt.figure(figsize=(12, 6))

# Depth Plot
plt.subplot(1, 2, 1)
plt.plot(depths_hill_climbing, label='Hill Climbing Depth', marker='o')
plt.plot(depths_iterative, label='Iterative Hill Climbing Depth', marker='o')
plt.plot(depths_vnd, label='Variable Neighbourhood Descent Depth', marker='o')
plt.title('Depth Explored')
plt.xlabel('Iteration')
plt.ylabel('Depth')
plt.legend()

# Nodes Visited Plot
plt.subplot(1, 2, 2)
plt.plot(nodes_hill_climbing, label='Hill Climbing Nodes Visited', marker='o')
plt.plot(nodes_iterative, label='Iterative Hill Climbing Nodes Visited', marker='o')
plt.plot(nodes_vnd, label='Variable Neighbourhood Descent Nodes Visited', marker='o')
plt.title('Nodes Visited')
plt.xlabel('Iteration')
plt.ylabel('Nodes Visited')
plt.legend()

plt.tight_layout()
plt.show()