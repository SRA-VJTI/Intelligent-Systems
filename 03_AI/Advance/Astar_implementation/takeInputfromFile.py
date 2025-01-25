import heapq
import math

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    open_list = []
    heapq.heappush(open_list, (0, start))
    
    g_costs = {start: 0}
    
    came_from = {start: None}
    
    while open_list:
        current_f_cost, current = heapq.heappop(open_list)
        
        if current == end:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for direction in neighbors:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_cost = g_costs[current] + 1
                
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + euclidean_heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_cost, neighbor))
                    came_from[neighbor] = current
    
    return None

def read_maze_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    maze = []
    start = end = None

    for line in lines:
        line = line.strip()
        if line.startswith('start:'):
            start = tuple(map(int, line.split(':')[1].split(',')))
        elif line.startswith('end:'):
            end = tuple(map(int, line.split(':')[1].split(',')))
        else:
            maze.append(list(map(int, line.split())))

    return maze, start, end

filename = 'maze.txt'
maze, start, end = read_maze_from_file(filename)
path = astar(maze, start, end)

if path:
    print("Shortest Path:", path)
else:
    print("No path found")