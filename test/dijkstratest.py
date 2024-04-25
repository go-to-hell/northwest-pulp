import heapq

def dijkstra(graph, start_node):
    # Initialize dictionaries to store minimum distances and visited nodes
    distances = {node: float('inf') for node in graph['nodes']}
    distances[start_node] = 0
    visited = set()
    
    # Initialize dictionary to store edges included in shortest path
    shortest_edges = {edge_id: False for edge_id in graph['edges']}
    
    # Priority queue (min heap) to store nodes and their distances
    pq = [(0, start_node)]
    
    while pq:
        # Pop the node with the smallest distance
        distance, current_node = heapq.heappop(pq)
        
        # Skip if node is already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Update distances and shortest edges for neighboring nodes
        for edge_id, edge in graph['edges'].items():
            if edge['origin'] == current_node:
                neighbor = edge['target']
                new_distance = distances[current_node] + edge['weight']
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    shortest_edges[edge_id] = True
                    heapq.heappush(pq, (new_distance, neighbor))
    
    return {'Nodes': distances, 'Edges': shortest_edges}

import heapq

def max_route(graph, start_node):
    # Initialize dictionaries to store maximum distances and visited nodes
    max_distances = {node: float('-inf') for node in graph['nodes']}
    max_distances[start_node] = 0
    visited = set()
    
    # Initialize dictionary to store edges included in maximum path
    max_edges = {edge_id: False for edge_id in graph['edges']}
    
    # Priority queue (min heap) to store nodes and their distances
    pq = [(0, start_node)]
    
    while pq:
        # Pop the node with the largest distance
        distance, current_node = heapq.heappop(pq)
        
        # Skip if node is already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Update maximum distances and maximum edges for neighboring nodes
        for edge_id, edge in graph['edges'].items():
            if edge['origin'] == current_node:
                neighbor = edge['target']
                new_distance = max_distances[current_node] + edge['weight']
                if new_distance > max_distances[neighbor]:
                    max_distances[neighbor] = new_distance
                    max_edges[edge_id] = True
                    heapq.heappush(pq, (-new_distance, neighbor))
    
    return {'Nodes': max_distances, 'Edges': max_edges}

# Example usage
""" graph = {
    'nodes': {'A': 'Node A', 'B': 'Node B', 'C': 'Node C', 'D': 'Node D'},
    'edges': {
        'U': {'weight': 2, 'origin': 'A', 'target': 'B'},
        'V': {'weight': 10, 'origin': 'A', 'target': 'C'},
        'W': {'weight': 3, 'origin': 'B', 'target': 'D'},
        'X': {'weight': 4, 'origin': 'C', 'target': 'D'},
        'Y': {'weight': 7, 'origin': 'B', 'target': 'C'},
        'Z': {'weight': 1, 'origin': 'C', 'target': 'B'},
    }
} """

graph = {
    'nodes': {'A': 'Node A', 'B': 'Node B', 'C': 'Node C', 'D': 'Node D', 'E': 'Node E'},
    'edges': {
        'V': {'weight': 10, 'origin': 'A', 'target': 'B'},
        'W': {'weight': 3, 'origin': 'A', 'target': 'C'},
        'X': {'weight': 4, 'origin': 'A', 'target': 'D'},
        'Y': {'weight': 7, 'origin': 'B', 'target': 'C'},
        'Z': {'weight': 1, 'origin': 'D', 'target': 'C'},
        'U': {'weight': 2, 'origin': 'C', 'target': 'E'},
        'T': {'weight': 5, 'origin': '5', 'target': 'E'},
    }
}
start_node = 'A'
result = max_route(graph, start_node)
print(result)
