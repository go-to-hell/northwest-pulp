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

paradas = {
    'Rio Seco': ['Azul'],
    'Plaza Libertad': ['Azul'],
    'UPEA': ['Azul'],
    'Plaza La Paz': ['Azul'],
    '16 de Julio': ['Azul', 'Plateada', 'Roja'],
    'Cementerio': ['Roja'],
    'Central': ['Roja', 'Naranja'],
    'Armentia': ['Naranja'],
    'Periférica': ['Naranja'],
    'Villarroel': ['Naranja', 'Blanco'],
    'Busch': ['Blanco', 'Cafe'],
    'Triangular': ['Blanco'],
    'Del Poeta': ['Blanco', 'Celeste'],
    'Libertador': ['Celeste', 'Amarillo', 'Verde'],
    'Las Villas': ['Cafe'],
    'El Prado': ['Celeste'],
    'Del Teatro': ['Celeste'],
    'Alto Obrajes': ['Verde'],
    'Obrajes': ['Verde'],
    'Irpavi': ['Verde'],
    'Sopocachi': ['Amarillo'],
    'Buenos Aires': ['Amarillo'],
    'Satelite': ['Amarillo', 'Plateada'],
    'Faro Murillo': ['Plateada', 'Morada'],
    '6 de Marzo': ['Morada'],
    'Obelisco': ['Morada'],
}

WATTAGE = 5  # KWatts


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
    prob = LpProblem("Transportation Problem",
                     LpMaximize if maximize else LpMinimize)

    # Create a list of tuples containing all the possible routes for transport
    Routes = [(w, s) for w in tp.Origins for s in tp.Targets]

    # A dictionary called 'route_vars' is created to contain the referenced variables (the routes)
    route_vars = LpVariable.dicts(
        "Route", (tp.Origins, tp.Targets), lowBound=0, cat=LpInteger)

    # The objective function is added to 'prob' first
    prob += (
        lpSum([route_vars[w][s] * tp.costs[tp.Origins.index(w)]
              [tp.Targets.index(s)] for (w, s) in Routes]),
        "Sum of Transporting Costs"
    )

    # The supply maximum constraints are added to prob for each supply node (warehouse)
    for w in tp.Origins:
        prob += lpSum([route_vars[w][s] for s in tp.Targets]
                      ) <= supply[w], "Sum of Products out of Warehouse %s" % w

    # The demand minimum constraints are added to prob for each demand node (store)
    for s in tp.Targets:
        prob += lpSum([route_vars[w][s] for w in tp.Origins]
                      ) >= demand[s], "Sum of Products into Store %s" % s

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
        G.add_edge(edge['source'], edge['target'],
                   weight=int(edge['label']), id=edge_id)
    return G


def create_data(G):
    edges = {edge[2]['id']: {"source": edge[0], "target": edge[1],
                             "label": str(edge[2]['weight'])} for edge in G.edges(data=True)}
    return {"edges": edges}


def find_spanning_tree(data, maximize=False):
    G = create_graph(data)
    T = nx.maximum_spanning_tree(
        G, weight='weight') if maximize else nx.minimum_spanning_tree(G, weight='weight')
    return create_data(T)


def create_paths(data):
    paths = {f"path{i+1}": {"edges": [edge]}
             for i, edge in enumerate(data['edges'].keys())}
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

    #print("Adjacency List: \n", adjacencyList)
    return adjacencyList


@app.post("/adjacency_list/")
async def adjacency_list(VEGraph: GraphData):
    return VerticesEdgesToAdjacencyList(VEGraph.dict())

## Dijktra's algorithm ##


def dijkstra(graph, startNode, maximize=False):
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
            result["targetPath"] = create_paths(
                {"edges": {edgeid: 0 for edgeid in dijkstraresult[endNode]["path"]}})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def disable_lines(graph, unavailable_lines):
    """
    Based on the `paradas` dictionary, disable the lines that are not available, by removing the edges that belong to the unavailable lines but not the nodes.
    """
    result = {}
    disabled_edges_ids = []

    for nodeid, node in graph.items():
        for neighborid, neighbor in node['neighbors'].items():
            is_in_list = False
            for source_line in paradas[node['name']]:
                for target_line in paradas[graph[neighborid]['name']]:
                    if source_line in unavailable_lines and source_line == target_line:
                        is_in_list = True
                        break

            if not is_in_list:
                if nodeid not in result:
                    result[nodeid] = {
                        "neighbors": {},
                        "name": node['name']
                    }
                result[nodeid]['neighbors'][neighborid] = neighbor
            else:
                disabled_edges_ids.append(neighbor['edgeId'])

    # Add all nodes that are not connected to any other node
    for nodeid, node in graph.items():
        if nodeid not in result:
            result[nodeid] = {
                "neighbors": {},
                "name": node['name']
            }

    return result, disabled_edges_ids


def secondsToTime(seconds):
    # print("seconds to time", seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def secondsToEnergy(seconds, energy_constraint):
    return WATTAGE / (seconds * energy_constraint)


def adjustForEnergy(graph, energy_constraint):
    """
    Using the formula A_time=W/(time*x) where:
    - A_time is the adjusted weight for the energy constraint
    - W is the wattage of the cable car
    - time is the time in seconds of the edge
    - x is the energy constraint
    Reworks the weights of the graph to consider the energy constraint
    """
    if energy_constraint is None:
        raise Exception("Energy constraint is required")
    if energy_constraint <= 0 or energy_constraint > 1:
        raise Exception("Energy constraint must be a value between 0 and 1")
    
    for nodeid, node in graph.items():
        for neighborid, neighbor in node['neighbors'].items():
            neighbor['label'] = secondsToEnergy(neighbor['label'], energy_constraint)


def adjustForMoney(graph, startNode, cardtype):
    """
    TODO: there are some narrow cases for which this doesn't work as properly, will fix tommorrow
    Adjust the weights of the graph to consider the cost of the cable car; following this rules:
    - If a station only belongs to a single line, the cost of its outgoing edges is 0
    - If a station belongs to multiple lines, the cost of its outgoing edges is given by the cards transfer cost
    - If the station is `Del Poeta` specifically, the cost of its outgoing edges is 0
    - If the station is the starting station, the cost of its outgoing edges is the base price of the card, this rule overrides the others
    """

    if cardtype not in card_mappings:
        raise Exception("Card type not found")

    card = card_mappings[cardtype]

    for nodeid, node in graph.items():
        for neighborid, neighbor in node['neighbors'].items():
            print("route from ", node['name'], " to ", graph[neighborid]['name'])
            if node['name'] == graph[startNode]['name']:
                print("has to pay card price")
                neighbor['label'] = card['price']
            elif node['name'] == "Del Poeta":
                print("is Del Poeta")
                neighbor['label'] = 0
            elif len(paradas[node['name']]) == 1:
                print("only belongs to one line")
                neighbor['label'] = 0
            else:
                print("belongs to multiple lines")
                neighbor['label'] = card['transfer']

def formatEdgeMoney(graph, edge, startNode, cardtype):
    if edge['label'] == "Disabled":
        return edge

    if edge['source'] == startNode:
        edge['label'] = f"${card_mappings[cardtype]['price']:.2f}"
    elif edge['target'] == "Del Poeta":
        edge['label'] = "$0.00"
    elif len(paradas[graph[edge['source']]['name']]) == 1:
        edge['label'] = "$0.00"
    else:
        edge['label'] = f"${card_mappings[cardtype]['transfer']:.2f}"
    return edge




@app.post("/dijkstra/telefericos")
async def dijkstra_algorithm_telefericos(
    VEGraph: GraphData,
    startNode: str,
    endNode: str,
    maximize: bool = False,
    cardtype: str = "regular",
    targetVariable: str = "time",  # time, money or energy
    energy_constraint: Optional[float] = None,
    disabledLines: List[str] = []
):
    """
    Calcula el camino más corto en un grafo utilizando el algoritmo de Dijkstra.
    
    Parámetros:
        - VEGraph (GraphData): Los datos del grafo que contienen los vértices y las aristas.
        - startNode (str): El id de nodo de inicio para el cálculo del camino más corto.
        - endNode (str): El id de nodo final para el cálculo del camino más corto.
        - maximize (bool, opcional): Indica si se debe maximizar la variable objetivo. Por defecto es False.
        - cardtype (str, opcional): El tipo de tarjeta a utilizar para los cálculos de costo. Por defecto es "regular". Puede ser cualquiera de ["regular", "student", "senior"].
        - targetVariable (str, opcional): La variable objetivo para el cálculo del camino más corto. Puede ser "time", "money" o "energy". Por defecto es "time".
        - energy_constraint (float, opcional): La restricción de energía para el cálculo del camino más corto. Solo aplicable si targetVariable es "energy". Por defecto es None. Debe ser un valor entre 0 y 1.
        - disabledLines (List[str], opcional): Una lista de líneas a deshabilitar en el grafo. Puede ser cualquiera de ["Azul", "Roja", "Plateada", "Naranja", "Blanco", "Cafe", "Celeste", "Amarillo", "Verde", "Morada"]. Por defecto es [].

    Retorna:
        - Un objeto JSON con los nodos:
            - La distancia óptima a ese nodo.
            - El camino óptimo a ese nodo.
        - Un objeto JSON con las aristas.
            - Formateado según la variable objetivo.
        - El valor óptimo de la variable objetivo.
        - El camino óptimo. Formateado para v-network-graph
        - Los ids de las aristas deshabilitadas.
    """
    try:
        print("Convert graph to adjacency list")
        graph = VerticesEdgesToAdjacencyList(VEGraph.dict())

        print("Disabling lines", disabledLines)
        graph, disabled_edges_ids = disable_lines(graph, disabledLines)
        print("Resulting graph: ", graph)

        if targetVariable == "energy":
            print("Adjusting for energy")
            adjustForEnergy(graph, energy_constraint)
            print("Adjusted graph: ", graph)
        elif targetVariable == "money":
            print("Adjusting for money")
            adjustForMoney(graph, startNode, cardtype)
            print("Adjusted graph: ", graph)

        print("Running Dijkstra")
        dijkstraresult = dijkstra(graph, startNode, maximize)
        # esta es la solución básica de Dijkstra
        result = {"nodes": dijkstraresult}
        result["edges"] = VEGraph.dict()["edges"]

        if targetVariable == "money":
            print("Converting to currency: Nodes")
            for nodeid, node in dijkstraresult.items():
                # cast numbers to currency
                if node["distance"] != -1:
                    node["distance"] = f"${node['distance']:.2f}"

            # format the edges values
            print("Converting to currency: Edges")
            for edgeid, edge in result["edges"].items():
                if edgeid in disabled_edges_ids:
                    edge["label"] = "Disabled"
                else:
                    edge = formatEdgeMoney(graph, edge, startNode, cardtype)

        if targetVariable == "time":
            print("Converting to time: Nodes")
            for nodeid, node in dijkstraresult.items():
                # cast seconds to MM:SS
                if node["distance"] != -1:
                    node["distance"] = secondsToTime(node["distance"]) 

            # format the edges values
            print("Converting to time: Edges")
            for edgeid, edge in result["edges"].items():
                if edgeid in disabled_edges_ids:
                    edge["label"] = "Disabled"
                else:
                    edge["label"] = secondsToTime(int(edge['label']))

        if targetVariable == "energy":
            print("Converting to energy: Nodes")
            for nodeid, node in dijkstraresult.items():
                # cast numbers to energy
                if node["distance"] != -1:
                    node["distance"] = f"{node['distance']:.2f} GW"

            # format the edges values
            print("Converting to energy: Edges")
            for edgeid, edge in result["edges"].items():
                if edgeid in disabled_edges_ids:
                    edge["label"] = "Disabled"
                else:
                    edge["label"] = f"{secondsToEnergy(int(edge['label']), energy_constraint):.2f} GW"

       
        # append optimal value to result
            result["optimalValue"] = dijkstraresult[endNode]['distance']  
        result["optimalPath"] = create_paths(
            {"edges": {edgeid: 0 for edgeid in dijkstraresult[endNode]["path"]}})
        result["disabledEdges"] = disabled_edges_ids

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
