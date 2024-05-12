from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

from pydantic import BaseModel
from pulp import *

import heapq

origins = [
    "http://localhost",
    "http://localhost:5173",
]

class TransportationProblem(BaseModel):
    Origins: list[str]
    Targets: list[str]
    supply: dict[str, int]
    demand: dict[str, int]
    costs: list[list[int]]

class GraphData(BaseModel):
    nodes: dict
    edges: dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Te conectaste correctamente a la API de Algoritmos.\nPara ver la API navegable entra al endpoint /docs."}

@app.post("/transportation/")
async def transportation_problem(tp: TransportationProblem, maximize: bool = False):
    
    # Define the supply
    supply = tp.supply
    
    # Define the demand
    demand = tp.demand
    
    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("Transportation Problem", LpMaximize if maximize else LpMinimize)
    
    # Create a list of tuples containing all the possible routes for transport
    Routes = [(w, s) for w in tp.Origins for s in tp.Targets]
    
    # A dictionary called 'route_vars' is created to contain the referenced variables (the routes)
    route_vars = LpVariable.dicts("Route", (tp.Origins, tp.Targets), lowBound = 0, cat = LpInteger)
    
    # The objective function is added to 'prob' first
    prob += (
        lpSum([route_vars[w][s] * tp.costs[tp.Origins.index(w)][tp.Targets.index(s)] for (w, s) in Routes]),
        "Sum of Transporting Costs"
    )
    
    # The supply maximum constraints are added to prob for each supply node (warehouse)
    for w in tp.Origins:
        prob += lpSum([route_vars[w][s] for s in tp.Targets]) <= supply[w], "Sum of Products out of Warehouse %s" % w
    
    # The demand minimum constraints are added to prob for each demand node (store)
    for s in tp.Targets:
        prob += lpSum([route_vars[w][s] for w in tp.Origins]) >= demand[s], "Sum of Products into Store %s" % s
    
    # The problem is solved using PuLP's choice of Solver
    prob.solve()
    
    return {
        "status": LpStatus[prob.status],
        "objective": value(prob.objective),
        "solution": {
            w: {
                s: route_vars[w][s].varValue
                for s in tp.Targets
            } for w in tp.Origins
        },
        "origins": tp.Origins,
        "targets": tp.Targets,
    }

### KRUSKAL MST ###

def create_graph(data):
    G = nx.Graph()
    for node in data['nodes'].values():
        G.add_node(node['id'])
    for edge_id, edge in data['edges'].items():
        G.add_edge(edge['source'], edge['target'], weight=int(edge['label']), id=edge_id)
    return G

def create_data(G):
    edges = {edge[2]['id']: {"source": edge[0], "target": edge[1], "label": str(edge[2]['weight'])} for edge in G.edges(data=True)}
    return {"edges": edges}

def find_spanning_tree(data, maximize=False):
    G = create_graph(data)
    T = nx.maximum_spanning_tree(G, weight='weight') if maximize else nx.minimum_spanning_tree(G, weight='weight')
    return create_data(T)

def create_paths(data):
    paths = {f"path{i+1}": {"edges": [edge]} for i, edge in enumerate(data['edges'].keys())}
    return paths


@app.post("/spanning_tree/")
async def spanning_tree(data: GraphData, maximize: bool = False):
    data_mst = find_spanning_tree(data.dict(), maximize)
    paths = create_paths(data_mst)
    return {"data_mst": data_mst, "paths": paths}


### DIJKSTRA SHORTEST PATH ###

def dijkstra(graph, start_node, end_node, maximize=False):
    return max_route(graph, start_node, end_node=end_node) if maximize else min_route(graph, start_node, end_node=end_node)
    
def min_route(graph, start_node, end_node=None):
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
            if edge['source'] == current_node:
                neighbor = edge['target']
                new_distance = distances[current_node] + edge['label']
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    #shortest_edges[edge_id] = True
                    heapq.heappush(pq, (new_distance, neighbor))

    # Set any remaining unvisited nodes to -1
    for node in graph['nodes']:
        if node not in visited:
            distances[node] = -1

    # If the end node is specified, calculate the shortest path from start to end
    print("Calculating path from", start_node, "to", end_node)
    print(distances[end_node])
    timeout = 20
    if end_node and distances[end_node] != -1:
        path = []
        current_node = end_node
        while current_node != start_node:
            timeout -= 1
            for edge_id, edge in graph['edges'].items():
                if edge['target'] != current_node or distances[edge['source']] == -1:
                    continue
                print("current_node", current_node)
                print("Checking edge", edge_id, "from", edge['source'], "to", edge['target'], "with label", edge['label'])
                timeout -= 1
                if timeout == 0:
                    raise Exception("Timeout")
                print("distances[current_node]", distances[current_node])
                print("distances[edge['source']]", distances[edge['source']])
                if distances[current_node] == distances[edge['source']] + edge['label']:
                    path.append(edge_id)
                    current_node = edge['source']
                    break
        path.reverse()
        for edge_id in path:
            shortest_edges[edge_id] = True
    
    return {'nodes': distances, 'edges': shortest_edges}

def max_route(graph, start_node, end_node=None):
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
            if edge['source'] == current_node:
                neighbor = edge['target']
                new_distance = max_distances[current_node] + edge['label']
                if new_distance > max_distances[neighbor]:
                    max_distances[neighbor] = new_distance
                    heapq.heappush(pq, (-new_distance, neighbor))

    # Set any remaining unvisited nodes to -1
    for node in graph['nodes']:
        if node not in visited:
            max_distances[node] = -1

    # If end node is specified, calculate the maximum path from start to end
    print("Calculating path from", start_node, "to", end_node)
    print(max_distances[end_node])  
    timeout = 1000
    if end_node and max_distances[end_node] != -1:
        path = []
        current_node = end_node
        while current_node != start_node:
            timeout -= 1
            for edge_id, edge in graph['edges'].items():
                if edge['target'] != current_node or max_distances[edge['source']] == -1:
                    continue
                print("current_node", current_node)
                print("Checking edge", edge_id, "from", edge['source'], "to", edge['target'], "with label", edge['label'])
                timeout -= 1
                if timeout == 0:
                    raise Exception("Timeout")
                print("max_distances[current_node]", max_distances[current_node])
                print("max_distances[edge['source']]", max_distances[edge['source']])
                if max_distances[current_node] == max_distances[edge['source']] + edge['label']:
                    path.append(edge_id)
                    current_node = edge['source']
                    break
        path.reverse()
        for edge_id in path:
            max_edges[edge_id] = True

    
    return {'nodes': max_distances, 'edges': max_edges}

@app.post("/dijkstra/")
async def dijkstra_shortest_path(data: GraphData, start_node: str, end_node: str, maximize: bool = False):
    print("3 Calculating path from", start_node, "to", end_node)
    result = dijkstra(data.dict(), start_node, end_node, maximize)
    # Filter edges mapped to true
    result['edges'] = {edge_id: edge for edge_id, edge in result['edges'].items() if edge}
    result["edges"] = create_paths(result)
    return result
