import matplotlib.pyplot as plt

# Read vectors from the text file
vectors = []
with open("output.txt", "r") as file:
   for line in file:
       vector = list(map(int, line.split()))
       vectors.append(vector)

# Unpack vectors for plotting
vec1, vec2, vec3, vec4 = vectors
print("vec1", vec1)
print("vec2", vec2)
print("vec3", vec3)
print("vec4", vec4)

# Plot the vectors
plt.figure(figsize=(12, 6))
plt.plot(vec1, label="BFS")
plt.plot(vec2, label="DFS")
plt.plot(vec3, label="H1")
plt.plot(vec4, label="H2")

# Adding labels and title
plt.xlabel('Problem Number')
plt.ylabel('Number of Iterations')
plt.title('Plot of Integer Vectors')
plt.legend()

# Set x-ticks for all values
plt.xticks(range(len(vec1)))

# Show plot
plt.show()