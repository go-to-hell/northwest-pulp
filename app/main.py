from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

from pydantic import BaseModel
from pulp import *

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
    return {"message": "Te conectaste correctamente a la API de transporte."}

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