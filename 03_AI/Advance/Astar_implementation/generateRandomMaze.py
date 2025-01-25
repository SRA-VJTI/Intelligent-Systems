import numpy as np
import heapq
import random
import cv2

# Heuristic function: Calculates Manhattan distance between two points (a and b).
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* pathfinding algorithm to find the shortest path in the maze.
def a_star(maze, start, goal):
    size = len(maze)  # Size of the maze
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movement directions (right, down, left, up)

    open_list = []  # Priority queue for nodes to explore
    heapq.heappush(open_list, (0, start))  # Add the start node to the open list

    came_from = {}  # Dictionary to reconstruct the path
    g_score = {start: 0}  # Cost from the start to each node
    f_score = {start: heuristic(start, goal)}  # Estimated total cost from start to goal

    visited_steps = []  # List of all visited nodes (for visualization)

    while open_list:
        # Get the node with the lowest f_score value
        current_f, current = heapq.heappop(open_list)

        # If the goal is reached, reconstruct and return the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()  # Reverse the path to start from the beginning
            return path, visited_steps

        # Explore all neighbors of the current node
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check if the neighbor is within bounds and walkable
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size and maze[neighbor] == 0:
                tentative_g_score = g_score[current] + 1  # Calculate the tentative g_score

                # If this path to the neighbor is better, update its scores
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))  # Add the neighbor to the open list
                    visited_steps.append(neighbor)  # Add to visited nodes for visualization

    # If no path is found, return None
    return None, visited_steps

# Initialize a grid with all walls (1).
def initialize_grid(size):
    grid = np.ones((size, size), dtype=int)  # Create a grid filled with 1s
    return grid

# Generate a random maze using depth-first search.
def generate_maze(grid, x, y, size):
    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]  # Directions for movement (two cells at a time)
    random.shuffle(directions)  # Randomize directions for maze generation

    for dx, dy in directions:
        nx, ny = x + dx, y + dy  # Calculate the new cell's coordinates

        # Check if the new cell is within bounds and unvisited
        if 0 <= nx < size and 0 <= ny < size and grid[nx, ny] == 1:
            grid[x + dx // 2, y + dy // 2] = 0  # Remove the wall between cells
            grid[nx, ny] = 0  # Mark the new cell as part of the path
            generate_maze(grid, nx, ny, size)  # Recursively generate the maze

# Wrapper function to create a complete maze.
def create_maze(size):
    grid = initialize_grid(size)  # Initialize the grid
    grid[1, 1] = 0  # Set the starting cell as walkable
    generate_maze(grid, 1, 1, size)  # Generate the maze
    grid[0, 1] = 0  # Create an entrance
    grid[size - 1, size - 1] = 0  # Create an exit
    return grid

# Draw a single frame for the maze visualization.
def draw_maze_frame(maze, visited_steps, path, frame_num):
    # Create a blank white frame
    frame = np.full((maze.shape[0] * 20, maze.shape[1] * 20, 3), 255, dtype=np.uint8)

    # Draw the maze grid
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            color = (0, 0, 0) if maze[x, y] == 1 else (255, 255, 255)  # Black for walls, white for paths
            frame[x * 20:(x + 1) * 20, y * 20:(y + 1) * 20] = color

    # Highlight visited steps in red (up to the current frame number)
    if frame_num < len(visited_steps):
        for (x, y) in visited_steps[:frame_num + 1]:
            if (x, y) in path:  # If part of the final path, highlight in red
                frame[x * 20:(x + 1) * 20, y * 20:(y + 1) * 20] = (0, 0, 255)

    # Highlight the final path in green once the search is complete
    if frame_num >= len(visited_steps) - 1:
        for (x, y) in path:
            frame[x * 20:(x + 1) * 20, y * 20:(y + 1) * 20] = (0, 255, 0)

    return frame

# Solve the maze and create a video of the solution process.
def solve_maze(maze):
    start = (0, 1)  # Starting point
    goal = (len(maze) - 1, len(maze) - 1)  # Goal point

    # Find the shortest path using A*
    path, visited_steps = a_star(maze, start, goal)

    if path:
        # Create a video writer to save the visualization
        height, width = maze.shape[0] * 20, maze.shape[1] * 20
        out = cv2.VideoWriter('maze_solution_fast.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        # Generate and save each frame
        for frame_num in range(len(visited_steps) + 50):
            frame = draw_maze_frame(maze, visited_steps, path, frame_num)
            out.write(frame)

        out.release()  # Finalize and save the video
        print("Maze solution video saved as maze_solution_fast.mp4")
    else:
        print("No path found")

# Main function
if __name__ == "__main__":
    size = 30  # Size of the maze (30x30 grid)
    maze = create_maze(size)  # Generate a random maze
    solve_maze(maze)  # Solve the maze and save the solution video