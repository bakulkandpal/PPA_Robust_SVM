import numpy as np
from pyomo.environ import *

D = 2000  # Weight for SVM
alpha = 0.75  # For CVaR

# Load data (price_high.mat and price_low.mat should be loaded separately)
# Make sure columns price_low_yearly and price_high_yearly are within data file.

day_to_use = 90
hours_to_use = day_to_use * 24

price_low_yearly2 = price_low_yearly[:hours_to_use]
price_high_yearly2 = price_high_yearly[:hours_to_use]

price_low_fin = price_low_yearly2[price_low_yearly2 != 0]
price_high_fin = price_high_yearly2[price_high_yearly2 != 0]

min_length = min(len(price_low_fin), len(price_high_fin))

if len(price_low_fin) > len(price_high_fin):
    top_min_elements = np.partition(price_low_fin, len(price_low_fin) - min_length)[:min_length]
    price_low_fin = price_low_fin[~np.isin(price_low_fin, top_min_elements)]
else:
    top_max_elements = np.partition(price_high_fin, min_length - len(price_high_fin))[:len(price_high_fin)]
    price_high_fin = price_high_fin[~np.isin(price_high_fin, top_max_elements)]

N = 2 * len(price_low_fin)  # Total number of samples
n_features = 2  # Number of features 2 (time and prices)

model = ConcreteModel()

# Define variables
model.w = Var(range(n_features))
model.b = Var()
model.z = Var()
model.xi = Var(range(N))

# Objective
model.objective = Objective(expr=0.5 * sum(model.w[i] ** 2 for i in range(n_features)) + D * model.z
                           + D / (N * (1 - alpha)) * sum(model.xi[i] for i in range(N)))

# Constraints
model.constraints = ConstraintList()
for i in range(N):
    model.constraints.add(expr=Y[i] * (sum(X[i, j] * model.w[j] for j in range(n_features)) + model.b)
                         + model.z >= 1 - model.xi[i])
    model.constraints.add(expr=model.xi[i] + model.z >= 0)
    model.constraints.add(expr=model.xi[i] >= 0)

solver = SolverFactory('gurobi')
results = solver.solve(model)

if results.solver.status != SolverStatus.ok:
    print("Solver reported an error:", results.solver.status)
else:
    w_opt = [model.w[i].value for i in range(n_features)]
    b_opt = model.b.value
    xi_opt = [model.xi[i].value for i in range(N)]
    z_opt = model.z.value

    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], s=20, c=Y, cmap=plt.cm.Paired)
    plt.hold(True)

    x1Grid, x2Grid = np.meshgrid(np.linspace(min(X[:, 0]), max(X[:, 0]), 50),
                                 np.linspace(min(X[:, 1]), max(X[:, 1]), 50))
    gridX = np.column_stack((x1Grid.ravel(), x2Grid.ravel()))
    scores = np.dot(gridX, w_opt) + b_opt

    plt.contour(x1Grid, x2Grid, scores.reshape(x1Grid.shape), levels=[0], colors='k')
    plt.contour(x1Grid, x2Grid, scores.reshape(x1Grid.shape), levels=[-1, 1], colors='k', linestyles='dashed')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('EEL-SVM with Hinge Loss Decision Boundary')
    plt.show()
