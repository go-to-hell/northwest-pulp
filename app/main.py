from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

from pydantic import BaseModel
from pulp import *

from collections import deque

origins = [
    "http://localhost",
    "http://localhost:5173",
]

card_mappings = {
    "regular": {
        "price": 2.0,
        "transfer": 1.0
    },
    "student": {
        "price": 1.5,
        "transfer": 1.0
    },
    "senior": {
        "price": 1.0,
        "transfer": 1.0
    }
}

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


def VerticesEdgesToAdjacencyList(VEGraph):
    """
    Convert a graph from a list of vertices and edges to an adjacency list
    Input object JSON
    {
        "nodes":{
            "nodeId":{
                "id": "nodeId",
                "name": "nodeName"
            },
            ...
        },
        "edges":{
            "edgeId":{
                "id": "edgeId",
                "source": "sourceNodeId",
                "target": "targetNodeId",
                "label": "edgeLabel"
            },
            ...
        }
    }

    Expected output object JSON
    {
        "nodeId":{
            "neighbors":{
                "neighborNodeId": {
                    "edgeId": "edgeId",
                    "label": "edgeLabel"
                },
                ...
            },
            "name": "nodeName"
        }
    }

    A neighbor is defined as a node that is reachable from the source node of an edge to the target node of that edge
    """

    adjacencyList = {}
    for nodeId, node in VEGraph['nodes'].items():
        adjacencyList[nodeId] = {
            "neighbors": {},
            "name": node['name']
        }
    
    for edgeId, edge in VEGraph['edges'].items():
        if type(edge['label']) == str:
            print("Warning: Edge label is a string, converting to int")
            edge['label'] = int(edge['label'])
        sourceNode = edge['source']
        targetNode = edge['target']
        adjacencyList[sourceNode]['neighbors'][targetNode] = {
            "edgeId": edgeId,
            "label": edge['label']
        }

    print("Adjacency List: \n", adjacencyList)
    return adjacencyList

@app.post("/adjacency_list/")
async def adjacency_list(VEGraph: GraphData):
    return VerticesEdgesToAdjacencyList(VEGraph.dict())

## Dijktra's algorithm ##

def dijkstra(graph, startNode, maximize = False):
    """
    Given a graph in Adjacency List format, a start node, and a maximize boolean value, calculate the shortest/longest path from the start node to all other nodes in the graph.

    Expected output format
    {
        "nodeId": {
            "distance": distance,
            "path": [path]
        },
        ...
    }
    """

    # Initialize the distance and path dictionaries
    if maximize:
        distance = {node: float('-infinity') for node in graph}
    else:
        distance = {node: float('infinity') for node in graph}
    distance[startNode] = 0
    path = {node: [] for node in graph}

    # Create a queue to keep track of the nodes that need to be visited
    queue = deque([startNode])
    timeout = 125
    node = None

    # While the queue is not empty
    while queue:
        print("current queue: ", queue)
        print("current node: ", node)
        timeout -= 1
        if timeout == 0:
            print("Timeout")
            raise Exception("Timeout")
        # Get the first node in the queue
        node = queue.popleft()

        
        

        # Get the neighbors of the node
        for neighbor, edge in graph[node]['neighbors'].items():
            # Calculate the new distance
            newDistance = distance[node] + edge['label']

            # If the new distance is shorter/longer than the current distance
            if (maximize and newDistance > distance[neighbor]) or (not maximize and newDistance < distance[neighbor]):
                print("current path to ", neighbor, " is ", path[neighbor])
                # To prevent an infinite loop in maximizations, if the path to a node contains the same edge, skip it
                if edge['edgeId'] in path[neighbor]:
                    continue
                # Update the distance
                distance[neighbor] = newDistance
                # Update the path
                path[neighbor] = path[node] + [edge['edgeId']]
                # Add the neighbor to the queue
                queue.append(neighbor)

    # Any remaining nodes in the queue are unreachable from the start node, their distance will be set to -1
    for nodeid, node in graph.items():
        if distance[nodeid] == float('infinity'):
            distance[nodeid] = -1

    # Return the distance and path dictionaries
    return {node: {"distance": distance[node], "path": path[node]} for node in graph}

@app.post("/dijkstra/")
async def dijkstra_algorithm(VEGraph: GraphData, startNode: str, endNode: Optional[str] = None, maximize: bool = False):
    try:
        graph = VerticesEdgesToAdjacencyList(VEGraph.dict())
        dijkstraresult = dijkstra(graph, startNode, maximize)
        result = {"nodes": dijkstraresult}
        if endNode:
            result["targetPath"] = create_paths({"edges": {edgeid: 0 for edgeid in dijkstraresult[endNode]["path"]}})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def calculate_prices(graph, cardtype):
    # TODO: A paritr del tipo de tarjeta se deben llenar los precios de los nodos, conmsiderando
    # las tarifas de transferencia y las excepciones en las líenas celeste y plateada
    pass

def secondsToTime(seconds):
    # TODO: Convertir los segundos a un string en formato MM:SS
    pass

@app.post("/dijkstra/telefericos")
async def dijkstra_algorithm_telefericos(
    VEGraph: GraphData, 
    startNode: str, 
    maximize: bool = False, 
    cardtype: str = "regular",
    timeOrMoney: str = "time"):
    """
    TODO: Esta función fue generada con IA, se debe completar el código para que funcione correctamente.
    Según el usuario requiera optimizar por tiempo o por dinero, se debe calcular el costo de los nodos.
    En el caso de costo por dinero se debe convertir el grafo a un grafo ponderado por el precio de la tarjeta. Esto es última prioridad
    En el caso de costo por tiempo es necesario al resultado de dijkstra convertir los segundos a un string en formato MM:SS
    En ambos casos se debe retornar adicionalmente el costo óptimo de la ruta
    """
    try:
        graph = VerticesEdgesToAdjacencyList(VEGraph.dict())
        dijkstraresult = dijkstra(graph, startNode, maximize)
        result = {"nodes": dijkstraresult} # esta es la solución básica de Dijkstra
        if timeOrMoney == "money":
            for nodeid, node in dijkstraresult.items():
                if node["distance"] != -1:
                    node["distance"] = node["distance"] * card_mappings[cardtype]["price"]
        if timeOrMoney == "time":
            for nodeid, node in dijkstraresult.items():
                if node["distance"] != -1:
                    node["distance"] = node["distance"] / card_mappings[cardtype]["transfer"]

        # TODO Añadir lo indicado a la variable result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

