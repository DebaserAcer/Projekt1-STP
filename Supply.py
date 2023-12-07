import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.index_tricks import c_
import pandas as pd
import time




#------------------------------------------------------------------------------
def CreateNetwork(links: pd.DataFrame, cost_per_km):
    # Trick to avoid shortcuts through centroids,
    # give origin nodes and destination nodes different names
    # assumes that the same is done with OD matrix, of course    
    links['from_id'] = links['from_id'].str.replace('2480A', 'origA')
    links['from_id'] = links['from_id'].str.replace('2480B', 'origB')
    links['from_id'] = links['from_id'].str.replace('2480C', 'origC')
    links['to_id'] = links['to_id'].str.replace('2480A', 'destA')
    links['to_id'] = links['to_id'].str.replace('2480B', 'destB')
    links['to_id'] = links['to_id'].str.replace('2480C', 'destC')
    
    # Add some columns we need in the algorithms
    links['cost'] = links.apply(lambda l: cost_per_km * l['length']/1000, axis=1)
    links['freeflow'] = links.apply(lambda l: l['length']/(l['speed'] / 3.6) / 60, axis=1)
    links['x'] = 0  
    links['weight'] = 0
    links['traveltime'] = 0

    G = nx.from_pandas_edgelist(links, source='from_id', target="to_id", create_using=nx.DiGraph(), edge_attr=True)
    return G, links


#------------------------------------------------------------------------------
# Volume delay function (Bureau of Public Roads, 1964)
# it is less used in practice today, but useful for teaching purposes
# functions used in practice are more complex and realistic 
# but add little to understanding the idea
def BPR(t0, Q, lanes, v, alpha, beta):
    x = v / (Q * lanes)
    t = t0 * (1 + alpha * pow(x,beta))
    return t


#------------------------------------------------------------------------------
# Compute travel times based on link data and volume delay function
# For use directly by shortest path algorithm.
# Assumes that edge_data is a dictionary
# {length: nnn, speed: nnn} where length is in kilometers and speed in km/h
# returns traveltime in minutes
def TravelTime(start, end, edge_data):

    l = edge_data['length']
    s = edge_data['speed']
    freeflow = edge_data['freeflow']
    Q = edge_data['Q']
    lanes = edge_data['lanes']
    v = edge_data['x']
    alpha = edge_data['alpha']
    beta = edge_data['beta']

    t = BPR(freeflow, Q, lanes, v, alpha, beta)

    return t

# Compute travel times based on link data and volume delay function
# Assumes that edge_data is a dictionary
# but for an arbitrary flow x
def TravelTimeX(edge_data, x):

    l = edge_data['length']
    s = edge_data['speed']
    freeflow = edge_data['freeflow']
    Q = edge_data['Q']
    lanes = edge_data['lanes']
    v = x
    alpha = edge_data['alpha']
    beta = edge_data['beta']

    t = BPR(freeflow, Q, lanes, v, alpha, beta)

    return t

#--------------------------------------------------------------------------

def AllOrNothingX(OD_demand : np.ndarray, G : nx.DiGraph, origs: dict, dests: dict):
    
    # put results in a copy of the network graph
    AoN = {}
    
    # sp - shortest paths, returns both costs and the corresponding paths
    sp = nx.all_pairs_dijkstra_path(G, weight='weight')
    for i, paths in sp:
        if i in origs:
            #print(f"AoN {i}")
            for j in paths:
                if j in dests:
                    path = tuple(paths[j])
                    pairs = [path[i: i + 2] for i in range(len(path)-1)]
                    o = origs[i]
                    d = dests[j]
                    demand = OD_demand[o,d]

                    for pair in pairs:
                        #(s,t) = pair
                        AoN[pair] = AoN.get(pair, 0) + demand

    return AoN

#--------------------------------------------------------------------------

def AllOrNothing(OD_demand : np.ndarray, G : nx.DiGraph, origs: dict, dests: dict):
    
    # put results in a copy of the network graph
    AoN = {}
    
    # sp - shortest paths, returns both costs and the corresponding paths
    
    for i in origs.keys():
        paths = nx.single_source_dijkstra_path(G, source=i, weight='weight')
        #print(f"AoN {i}")
        for j in paths:
            if j in dests:
                path = tuple(paths[j])
                pairs = [path[i: i + 2] for i in range(len(path)-1)]
                o = origs[i]
                d = dests[j]
                demand = OD_demand[o,d]

                for pair in pairs:
                    #(s,t) = pair
                    AoN[pair] = AoN.get(pair, 0) + demand

    return AoN


#------------------------------------------------------------------------------
# Find the lambda that minimises the Beckman function
# See https://sboyles.github.io/teaching/ce392c/5-beckmannmsafw.pdf for details
# Basically searching for a point between x and x* using bisection
def FindLambda(G, x, x_aon):
    w_lo = 0
    w_hi = 1

    while w_hi - w_lo > 0.001:
        w = (w_lo + w_hi) / 2
        x_hat = w * x_aon + (1-w) * x
        t_wsum = 0
        l = 0
        t_l = np.zeros(len(x))
        for s,t in G.edges:
            t_l[l] = TravelTimeX(G.edges[s,t], x_hat[l])
            t_wsum += t_l[l] * (x_aon[l] - x[l])
            l += 1

        if t_wsum > 0:
            w_hi = w
        else:
            w_lo = w

    return w
            

def Skim(G, origs, dests):
    # Fixed traveltimes on the diagonal
    nz = len(origs)
    t_ij = np.eye(nz) * 3
    c_ij = np.eye(nz) * 0.6
    d_ij = np.eye(nz) * 0.3
    sp_time = 0
    skim_time = 0
    for i in origs.keys():
        start_sp = time.time()
        paths = nx.single_source_dijkstra_path(G, source=i, weight='weight')
        sp_time += time.time() - start_sp
        #print(f"AoN {i}")
        start_skim = time.time()
        for j in paths:
            if j in dests:
                o = origs[i]
                d = dests[j]
                path = tuple(paths[j])
                pairs = [path[i: i + 2] for i in range(len(path)-1)]
                for pair in pairs:
                    (s,t) = pair
                    t_ij[o,d] += G.edges[s,t]['traveltime']
                    c_ij[o,d] += G.edges[s,t]['cost']
                    d_ij[o,d] += G.edges[s,t]['length']
        skim_time += time.time() - start_skim
    #print('sp ', sp_time)
    #print('skim', skim_time)
    return (t_ij,c_ij,d_ij/1000)

#--------------------------------------------------------------------------
# Solves the traffic assignment problem
# Using either MSA or Frank-Wolfe, selected by passing
# 'msa' or 'fw' as the last parameter
# The only difference is whether lambda is set to 1/k 
# or searched for with the function above 
def Assignment(OD_demand  : np.ndarray, G_base : nx.DiGraph, gap: float, iters: int, method, origs, dests):
    k = 1
    G = G_base.copy()
    VoT = 116 / 60
    print(f'{method}-assignment: Create start solution')
    # Find a feasible starting solution
    # first update the weights on each link
    for s,t in G.edges:
        ttime = TravelTime(s,t,G.edges[s,t])
        G.edges[s,t]['traveltime'] = ttime
        G.edges[s,t]['weight'] = ttime * VoT + G.edges[s,t]['cost']
    
    # then assign all flow on shortest path
    aon = AllOrNothing(OD_demand, G_base, origs, dests)
    
    # set the link flows in the graph
    for s,t in G.edges:
        if (s,t) in aon:
            x = aon[(s,t)]
        else:
            x = 0
        G.edges[s,t]['x'] = x
    
    sumT = sum(sum(OD_demand))

    diff = 100
    relgap = 1
    while relgap > gap and k < iters:

        #print(f"Iteration {k}")

        # some vectors to keep data in
        diffs = np.zeros(len(G.edges))
        x = np.zeros(len(G.edges))
        x_aon = np.zeros(len(G.edges))
        times = np.zeros(len(G.edges))

 
 #       if k%10 == 0:

        
        # update weights for the current solution
        l = 0
        for s,t in G.edges:
            times[l] = TravelTime(s, t, G.edges[s,t])
            G.edges[s,t]['traveltime'] = times[l]
            G.edges[s,t]['weight'] = times[l] + G.edges[s,t]['cost']
            # save the current solution in a vector
            x[l] = G.edges[s,t]['x']
            l += 1

        # assign all flow on shortest paths
        aon = AllOrNothing(OD_demand, G, origs, dests)

        l = 0
        for s,t in G.edges:
            # put the all-or-nothing solution in a vector
            if (s,t) in aon:
                x_aon[l] = aon[(s,t)]
            l += 1
        
        l = 0
        # choose lambda (step size)
        if method == 'fw':
            w = FindLambda(G, x, x_aon)
        elif method == 'msa':
            w = 1/k
        else:
            print("choose method = 'msa' or 'fw'")
            break

        for s,t in G.edges:
            # update the current solution by stepping towards x*
            G.edges[s,t]['x'] = (1 - w) *  x[l] + w * x_aon[l]
            l += 1
        
        # Total System Travel Time
        TSTT = sum(x * times)

        # Shortest Path Travel Times
        # i.e. current travel times, shortest path (aon) flows
        SPTT = sum(x_aon * times)

        #print(f"TSTT: {round(TSTT,2)}, SPTT: {round(SPTT,2)}")

        diffs = x - x_aon
        diff = np.linalg.norm(diffs, 1)
        relgap = (TSTT / SPTT) - 1
        AEC = (TSTT - SPTT) / sumT
        #if k%10 == 0:
        print(f"Iteration: {k}, relgap: {round(relgap,6)}") 
        #    print("AEC     ", round(AEC, 6))
        #    print("diff    ", round(diff,2))
        #    print("time ", *np.round(times,1))
        #    print("x    ", *np.round(x,1))
        #    print("x_aon", *np.round(x_aon,1))        
        k += 1
    print(f"  {method} assignment: k = {k} relgap = {round(relgap,6)}, AEC = {round(AEC, 6)}")
    return G


