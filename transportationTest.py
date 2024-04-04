from pulp import *

# List all supply nodes
Warehouses = ["A", "B", "C"]

# List all demand nodes
Stores = ["W", "X", "Y", "Z"]

# Define the supply
supply = {
    "A": 30,
    "B": 50,
    "C": 30
}

# Define the demand
demand = {
    "W": 10,
    "X": 40,
    "Y": 20,
    "Z": 40
}

# Define the cost of transportation
costs = [
    #W   X   Y   Z
    [10, 10, 40, 10], # A
    [40, 10, 30, 20], # B
    [60, 30, 20, 10]  # C
]

# Create the 'prob' variable to contain the problem data
prob = LpProblem("Transportation Problem", LpMinimize)

# Create a list of tuples containing all the possible routes for transport
Routes = [(w, s) for w in Warehouses for s in Stores]

# A dictionary called 'route_vars' is created to contain the referenced variables (the routes)
route_vars = LpVariable.dicts("Route", (Warehouses, Stores), lowBound = 0, cat = LpInteger)

# The objective function is added to 'prob' first
prob += (
    lpSum([route_vars[w][s] * costs[Warehouses.index(w)][Stores.index(s)] for (w, s) in Routes]),
    "Sum of Transporting Costs"
)

# The supply maximum constraints are added to prob for each supply node (warehouse)
for w in Warehouses:
    prob += lpSum([route_vars[w][s] for s in Stores]) <= supply[w], "Sum of Products out of Warehouse %s" % w

# The demand minimum constraints are added to prob for each demand node (store)
for s in Stores:
    prob += lpSum([route_vars[w][s] for w in Warehouses]) >= demand[s], "Sum of Products into Store %s" % s

# The problem data is written to an .lp file
prob.writeLP("TransportationProblem.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen
print("Total Cost of Transportation = ", value(prob.objective))

# End of transportationTest.py