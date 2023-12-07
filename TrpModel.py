import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Supply import *
from Demand import *


#------------------------------------------------------------------------------
# The full four step model
# Input variables:
# 
# Output variables:
#
def FourStepModel(G: nx.DiGraph, landuse: pd.DataFrame, workers, jobs, iterations):

    # Some settings
    gap = 10e-4     # Stopping criterium for assignment solution 
    iters = 100     # Max iterations in assignment
    method = 'fw'   # uses Frank-Wolfe algorithm, 'msa' also possible
    lu = landuse.copy()
    lu['from_id'] = lu['deso']
    lu['from_id'] = lu['from_id'].str.replace('2480A', 'origA')
    lu['from_id'] = lu['from_id'].str.replace('2480B', 'origB')
    lu['from_id'] = lu['from_id'].str.replace('2480C', 'origC')
    lu['to_id'] = lu['deso']
    lu['to_id'] = lu['to_id'].str.replace('2480A', 'destA')
    lu['to_id'] = lu['to_id'].str.replace('2480B', 'destB')
    lu['to_id'] = lu['to_id'].str.replace('2480C', 'destC')

    orig_order = dict(zip(lu.from_id,lu.index))
    dest_order = dict(zip(lu.to_id,lu.index))

    n_zones = len(landuse)
    demand = np.ones((n_zones,n_zones))
    print(f'============== Computing free flow conditions ==============')
    G_start = Assignment(demand, G, 1e-2, 10,'fw',orig_order, dest_order)
    t_ij,c_ij,d_ij = Skim(G_start, orig_order, dest_order)
    T_ij_old = np.zeros((n_zones,n_zones))
    for iter in range(5):
        print(f'============== Outer iteration {iter} ==============')
        print('Updating demand')
        w = 1.0/(iter + 1)
        T_ij = w * Demand(t_ij, c_ij, lu, workers, jobs) + (1-w)*T_ij_old
        diffnorm = np.linalg.norm(T_ij_old - T_ij)
        print(f"norm(T - T_old): {diffnorm}")

        T_ij_old = T_ij

        print('Assigning demand')
        G_next = Assignment(T_ij, G, 1e-2, 10,'fw',orig_order, dest_order)

        print('Skimming time, cost, distance')
        t_ij,c_ij,d_ij = Skim(G_next, orig_order, dest_order)
        if diffnorm < 1:
            break
    return G_next, T_ij, t_ij, d_ij
