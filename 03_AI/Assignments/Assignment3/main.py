import random
import matplotlib.pyplot as plt

nodes = 0  # Global variable to track number of nodes visited

def random_candidate(m, n):  # Generate random binary assignment for variables
    return [random.randint(0, 1) for _ in range(n + 1)]

def generate_k_sat(k, m, n):  # Create a random k-SAT problem with m clauses and n variables
    k_sat = []
    for i in range(m):
        clause = set()
        while len(clause) < k:
            var = random.randint(1, n)
            if random.random() > 0.5:
                var = -var
            clause.add(var)
        k_sat.append(list(clause))
    return k_sat

def evaluate(k_sat, assignment):  # Count the number of satisfied clauses in the k-SAT problem
    count = 0
    for clause in k_sat:
        if any((assignment[abs(x)] == (x > 0)) for x in clause):
            count += 1
    return count

def move_gen(k_sat, node, n, bits=1):  # Generate neighboring solutions by flipping 1-3 bits
    newnodes = []
    if bits == 1:
        for i in range(1, n + 1):
            temp = node[:]
            temp[i] = 1 - temp[i]
            newnodes.append((evaluate(k_sat, temp), temp))
    elif bits == 2:
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                temp = node[:]
                temp[i] = 1 - temp[i]
                temp[j] = 1 - temp[j]
                newnodes.append((evaluate(k_sat, temp), temp))
    elif bits == 3:
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                for k in range(j + 1, n + 1):
                    temp = node[:]
                    temp[i] = 1 - temp[i]
                    temp[j] = 1 - temp[j]
                    temp[k] = 1 - temp[k]
                    newnodes.append((evaluate(k_sat, temp), temp))
    return newnodes

def hill_climbing(k_sat, n, m):  # Basic hill climbing algorithm to find local maxima
    global nodes
    node = random_candidate(m, n)
    depth = 0
    newnodes = move_gen(k_sat, node, n)
    newnode = max(newnodes, key=lambda x: x[0])[1]
    while evaluate(k_sat, newnode) > evaluate(k_sat, node):
        node = newnode
        newnodes = move_gen(k_sat, node, n)
        newnode = max(newnodes, key=lambda x: x[0])[1]
        depth += 1
        nodes += 1
    return depth, newnode

def iterative_hill_climbing(k_sat, n, m, iterations=100):  # Repeat hill climbing multiple times to improve solution
    global nodes
    best_node = random_candidate(m, n)
    total_depth = 0
    for _ in range(iterations):
        depth, node = hill_climbing(k_sat, n, m)
        total_depth += depth
        if evaluate(k_sat, node) > evaluate(k_sat, best_node):
            best_node = node
    return total_depth // iterations, best_node

def variable_neighborhood_descent(k_sat, n, m):  # Advanced search strategy exploring different neighborhood sizes
    global nodes
    node = random_candidate(m, n)
    best_node = node
    total_nodes_visited = 0
    total_depth = 0
    for bits in range(1, 4):
        newnodes = move_gen(k_sat, node, n, bits)
        total_nodes_visited += len(newnodes)
        newnode = max(newnodes, key=lambda x: x[0])[1]
        while evaluate(k_sat, newnode) > evaluate(k_sat, node):
            node = newnode
            newnodes = move_gen(k_sat, node, n, bits)
            total_nodes_visited += len(newnodes)
            newnode = max(newnodes, key=lambda x: x[0])[1]
            total_depth += 1
            nodes += 1
        if evaluate(k_sat, node) > evaluate(k_sat, best_node):
            best_node = node
    return (total_nodes_visited, total_depth), best_node

def main():  # Main function to run experiments and visualize results
    k = int(input("Enter the number of literals per clause: "))
    n = int(input("Enter the number of variables: "))
    iterations = 10
    
    # Lists to store performance metrics for different search strategies
    depths_hill_climbing, nodes_hill_climbing = [], []
    depths_iterative, nodes_iterative = [], []
    depths_vnd, nodes_vnd = [], []
    
    # Run experiments for different numbers of clauses
    for i in range(1, iterations + 1):
        m = i
        k_sat = generate_k_sat(k, m, n)
        print(f'The generated K-SAT for {m} clauses and {n} literals is {k_sat}')
        global nodes
        nodes = 0
        depth_hc, _ = hill_climbing(k_sat, n, m)
        depths_hill_climbing.append(depth_hc)
        nodes_hill_climbing.append(nodes)
        
        nodes = 0
        depth_iter, _ = iterative_hill_climbing(k_sat, n, m)
        depths_iterative.append(depth_iter)
        nodes_iterative.append(nodes)
        
        nodes = 0
        (nodes_v, depth_v), _ = variable_neighborhood_descent(k_sat, n, m)
        depths_vnd.append(depth_v)
        nodes_vnd.append(nodes_v)
    
    # Plotting results to compare search strategies
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(depths_hill_climbing, label='Hill Climbing Depth', marker='o')
    plt.plot(depths_iterative, label='Iterative Hill Climbing Depth', marker='o')
    plt.plot(depths_vnd, label='Variable Neighborhood Descent Depth', marker='o')
    plt.title('Depth Explored')
    plt.xlabel('Iteration')
    plt.ylabel('Depth')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(nodes_hill_climbing, label='Hill Climbing Nodes Visited', marker='o')
    plt.plot(nodes_iterative, label='Iterative Hill Climbing Nodes Visited', marker='o')
    plt.plot(nodes_vnd, label='Variable Neighborhood Descent Nodes Visited', marker='o')
    plt.title('Nodes Visited')
    plt.xlabel('Iteration')
    plt.ylabel('Nodes Visited')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()