from fastapi import FastAPI
from pydantic import BaseModel
from pulp import *

class TransportationProblem(BaseModel):
    Origins: list[str]
    Targets: list[str]
    supply: dict[str, int]
    demand: dict[str, int]
    costs: list[list[int]]

app = FastAPI()

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
        }
    }