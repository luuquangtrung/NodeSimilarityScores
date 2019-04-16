from scipy.optimize import linprog
import networkx as nx
from networkx import utils
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected
import numpy as np
import time

def lp_without_maxs_or_goal(N, SPG, SPG_prime, travel_min, travel_max):
    A = []
    b = []
    for i in range(0, N):
        for j in range(i + 1, N):
            for k in range(0, N):
                for l in range(k + 1, N):
                    if SPG[i][j] >= travel_min and SPG_prime[k][l] >= travel_min and SPG[i][j] <= travel_max and SPG_prime[k][l] <= travel_max:
                        new_constraint = [0 for n in range(0, N * N)]  # w_ik and w_jl
                        new_constraint[N * i + k] = -1
                        new_constraint[N * j + l] = -1
                        A.append(new_constraint)
                        b.append(-abs(SPG[i][j] - SPG_prime[k][l]))
                        new_constraint = [0 for n in range(0, N * N)]  # w_il and w_jk
                        new_constraint[N * i + l] = -1
                        new_constraint[N * j + k] = -1
                        A.append(new_constraint)
                        b.append(-abs(SPG[i][j] - SPG_prime[k][l]))
    return (A, b)

def maxs_matrix(N):
    A = []
    for i in range(0, N):
        new_constraint = [1 if N*i <= j < N * (i+1) else 0 for j in range(0, N*N)]
        A.append(new_constraint)
    for i in range(0, N):
        new_constraint = [1 if j % N == i else 0 for j in range(0, N * N)]
        A.append(new_constraint)
    return A

def goal_vector(N, target_min=None):
    c = [1 for n in range(0, N * N)]
    if target_min is not None:  # If there's one node (in G, not G_prime) we really want to minimize
        for i in range(0, N):
            c[N * target_min + i] += N
    return c

def get_values(G, G_prime):
    N = len(G.nodes())
    #print('Finding sortest paths...')
    SPG = dict(all_pairs_shortest_path_length(G))
    SPG_prime = dict(all_pairs_shortest_path_length(G_prime))
    #print('Found sortest paths.')

    #print('Generating LP...')
    (AGGP, bGGP) = lp_without_maxs_or_goal(N, SPG, SPG_prime, 1, N - 1)

    #print('Generated LP.')
    c = goal_vector(N)
    #print('Solving LP...')
    G_with_G_prime = linprog(c, A_ub=AGGP, b_ub=bGGP, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})
    #print('Solved LP...')
    return G_with_G_prime.x

def permute_labels_only(G):
    N = len(G.nodes())
    permutation = np.random.permutation([i for i in range(0, N)])
    G_prime = nx.Graph()
    for i in range(0, N):
        G_prime.add_node(i)
    for edge in G.edges():
        G_prime.add_edge(permutation[edge[0]], permutation[edge[1]])
    return G_prime

def make_graph_with_same_degree_dist(G):
    G_sequence = list(d for n, d in G.degree())
    G_sequence.sort()
    sorted_G_sequence = list((d, n) for n, d in G.degree())
    sorted_G_sequence.sort(key=lambda tup: tup[0])
    done = False
    while not done:
        G_prime = nx.configuration_model(G_sequence)
        G_prime = nx.Graph(G_prime)
        G_prime.remove_edges_from(G_prime.selfloop_edges())
        tries = 10
        while tries > 0 and (len(G.edges()) != len(G_prime.edges())):
            sorted_G_prime_sequence = list((d, n) for n, d in G_prime.degree())
            sorted_G_prime_sequence.sort(key=lambda tup: tup[0])
            #print("Sorted G_sequence:")
            #print(sorted_G_sequence)
            #print("Sorted G_prime_sequence:")
            #print(sorted_G_prime_sequence)
            missing = []
            for i in range(0, len(G.nodes())):
                while sorted_G_sequence[i][0] > sorted_G_prime_sequence[i][0]:
                    missing.append(sorted_G_prime_sequence[i][1])
                    sorted_G_prime_sequence[i] = (sorted_G_prime_sequence[i][0] + 1, sorted_G_prime_sequence[i][1])
            missing = np.random.permutation(missing)
            if len(missing) % 2 != 0:
                print("Sanity issue! Alert!")
            #print("Edges before:")
            #print(G_prime.edges())
            #print("Missing:")
            #print(missing)
            for i in range(0, int(len(missing) / 2)):
                G_prime.add_edge(missing[2*i], missing[2*i + 1])
            G_prime = nx.Graph(G_prime)
            G_prime.remove_edges_from(G_prime.selfloop_edges())
            #print("Edges after:")
            #print(G_prime.edges())
            if not is_connected(G_prime):
                # print("Bad: G_prime disconnected")
                pass
            tries -= 1
        if not is_connected(G_prime):
            pass
        elif len(G.edges()) == len(G_prime.edges()):
            #print("Graph creation successful")
            done = True
    return G_prime

for size in range(16,32):
    #print("Creating Pairs of Graphs")
    good = False
    while not good:
        # Generate first G
        using_sequence = False
        #sequence = [2, 2, 2, 2, 6, 4, 4, 4, 4]  # Set sequence
        #G=nx.configuration_model(sequence)

        G=nx.erdos_renyi_graph(size,0.4)
        #G=nx.watts_strogatz_graph(10,3,0.3)
        #G=nx.barabasi_albert_graph(10,2)

        G=nx.Graph(G)
        G.remove_edges_from(G.selfloop_edges())
        if not is_connected(G):
            # print("Bad: G disconnected")
            continue
        good = True
        G_prime = make_graph_with_same_degree_dist(G)
        # G_prime = permute_labels_only(G)

    start_time = time.time()
    numbers = get_values(G, G_prime)
    end_time = time.time()
    # print(numbers)
    print("%s, %s" % (size, end_time - start_time))

    # Get actual result
    GM = isomorphism.GraphMatcher(G, G_prime)
    actual_iso = GM.is_isomorphic()
