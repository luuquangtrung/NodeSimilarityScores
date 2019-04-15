from scipy.optimize import linprog
import networkx as nx
from networkx import utils
from networkx.algorithms.bipartite.generators import configuration_model
from networkx.algorithms import isomorphism
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.components import is_connected
import numpy as np

# IMPORTANT: By default bounds on a variable are 0 <= v < +inf

# c = [-1, 4]
# A = [[-3, 1], [1, 2], [0, -1]]
# b = [6, 4, 3]
# res = linprog(c, A_ub=A, b_ub=b, options={"disp": True})
# print(res)

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
                print("Bad: G_prime disconnected")
            tries -= 1
        if not is_connected(G_prime):
            pass
        elif len(G.edges()) == len(G_prime.edges()):
            #print("Graph creation successful")
            done = True
    return G_prime

def lp_without_maxs_or_goal(N, SPG, SPG_prime, travel_min):
    A = []
    b = []
    for i in range(0, N):
        for j in range(i + 1, N):
            for k in range(0, N):
                for l in range(k + 1, N):
                    if SPG[i][j] >= travel_min and SPG_prime[k][l] >= travel_min:
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

def goal_vector_prime(N, target_min=None):
    c = [1 for n in range(0, N * N)]
    if target_min is not None:  # If there's one node (in G, not G_prime) we really want to minimize
        for i in range(0, N):
            c[N * i + target_min] += N
    return c

def weight_caps(N, result):
    caps = [0.01 for n in range(0, N)]  # TODO: The 0.01 is for slack. It should probably be a function of N.
    for i in range(0, N * N):
        caps[int(i / N)] += result.x[i]
    return caps

def weight_caps_prime(N, result):
    caps = [0.01 for n in range(0, N)]  # TODO: The 0.01 is for slack. It should probably be a function of N.
    for i in range(0, N * N):
        caps[i % N] += result.x[i]
    return caps

def check_for_min_travel(N, SPG, SPG_prime, min_travel):
    # First run the graphs against themselves
    (AG, bG) = lp_without_maxs_or_goal(N, SPG, SPG, min_travel)
    c = goal_vector(N)
    G_with_G = linprog(c, A_ub=AG, b_ub=bG, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    (AG_prime, bG_prime) = lp_without_maxs_or_goal(N, SPG_prime, SPG_prime, min_travel)
    G_prime_with_G_prime = linprog(c, A_ub=AG_prime, b_ub=bG_prime, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    # First check: Are sums not equal?
    if abs(G_with_G.fun - G_prime_with_G_prime.fun) > 0.01:  # TODO: Replace this constant.
        return False
    print(G_with_G.fun)
    print(G_prime_with_G_prime.fun)

    print ("Going on to second round test.")
    # Then run them against each other, holding G's caps from before:
    (AGGP, bGGP) = lp_without_maxs_or_goal(N, SPG, SPG_prime, min_travel)
    A_weight_caps = maxs_matrix(N)
    A = AGGP + A_weight_caps
    b_G1_caps = weight_caps(N, G_with_G)
    b_G2_caps = [N*N for n in range(0, N)]
    b = bGGP + b_G1_caps + b_G2_caps
    G_with_G_prime = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    # Second check: Are sums not equal or did it fail??
    if G_with_G_prime.status != 0 or abs(G_with_G.fun - G_with_G_prime.fun) > 0.01:  # TODO: Replace this constant.
        print("Second check failed.")
        #print(G_with_G_prime)
        return False

    # Use G_with_G_prime to get caps for G_prime, then see if it can use those with itself.
    A = AG_prime + A_weight_caps
    b_G1_caps = weight_caps_prime(N, G_with_G_prime)
    b = bG_prime + b_G1_caps + b_G2_caps
    G_prime_with_G_prime = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})

    # Third check: Can G_prime match with itself using the limits from G_with_G_prime?
    if G_prime_with_G_prime.status != 0 or abs(G_with_G.fun - G_prime_with_G_prime.fun) > 0.01:  # TODO: Replace this constant.
        print("Third check failed.")
        return False

    return True #TODO: Remove this line.

    # Lastly, do the full operation.
    for i in range(0, N):
        # Measure G with itself where goal is to minimize sum_j w_ij
        c = goal_vector(N, i)
        A = AG + A_weight_caps
        b = bG + [N*N for n in range(0, 2*N)]
        G_with_G = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})
        A = AGGP + A_weight_caps
        b_G1_caps = weight_caps(N, G_with_G)
        print(b_G1_caps[i])
        b = bGGP + b_G1_caps + b_G2_caps
        G_with_G_prime = linprog(c, A_ub=A, b_ub=b, method="interior-point", options={"disp":False, "maxiter":N*N*N*N*100})
        if G_with_G_prime.status != 0 or abs(G_with_G.fun - G_with_G_prime.fun) > 0.01:
            print("Fourth check failed.")
            return False

    # Need to add the reverse direction.

    return True

def iso_check(G, G_prime):
    N = len(G.nodes())
    SPG = dict(all_pairs_shortest_path_length(G))
    SPG_prime = dict(all_pairs_shortest_path_length(G_prime))
    longest_sp = 0
    for i in range(0, N):
        for j in range(i + 1, N):
            if SPG[i][j] > longest_sp:
                longest_sp = SPG[i][j]
            if SPG_prime[i][j] > longest_sp:
                longest_sp = SPG_prime[i][j]
    for min_travel in range(0, longest_sp + 1):
        print("Round " + str(min_travel))
        if not check_for_min_travel(N, SPG, SPG_prime, min_travel):
            return False
    return True

def old_counter_example():
    G = nx.Graph()
    G_prime = nx.Graph()
    for i in range(0, 8):
        G.add_node(i)
        G_prime.add_node(i)
    for G_edge in [(0, 2), (1, 3), (1, 4), (1, 7), (2, 3), (2, 5), (2, 6), (3, 5), (3, 7), (4, 6), (5, 6)]:
        G.add_edge(G_edge[0], G_edge[1])
    for G_prime_edge in [(0, 3), (1, 5), (1, 7), (2, 4), (2, 6), (3, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]:
        G_prime.add_edge(G_prime_edge[0], G_prime_edge[1])
    return (G, G_prime)

def permute_labels_only(G):
    N = len(G.nodes())
    permutation = np.random.permutation([i for i in range(0, N)])
    G_prime = nx.Graph()
    for i in range(0, N):
        G_prime.add_node(i)
    for edge in G.edges():
        G_prime.add_edge(permutation[edge[0]], permutation[edge[1]])
    return G_prime

for i in range(1,15):
    #print("Creating Pairs of Graphs")
    good = False
    while not good:
        # Generate first G
        using_sequence = False
        #sequence = [2, 2, 2, 2, 6, 4, 4, 4, 4]  # Set sequence
        #G=nx.configuration_model(sequence)

        G=nx.erdos_renyi_graph(8,0.4)
        #G=nx.watts_strogatz_graph(10,3,0.3)
        #G=nx.barabasi_albert_graph(10,2)

        G=nx.Graph(G)
        G.remove_edges_from(G.selfloop_edges())
        if not is_connected(G):
            print("Bad: G disconnected")
            continue
        good = True
        G_prime = make_graph_with_same_degree_dist(G)
        # G_prime = permute_labels_only(G)

    (G, G_prime) = old_counter_example()
    #print(G.edges())
    #print(G_prime.edges())
    #G_prime = permute_labels_only(G_prime)
    predict_iso = iso_check(G, G_prime)

    # Get actual result
    GM = isomorphism.GraphMatcher(G, G_prime)
    actual_iso = GM.is_isomorphic()

    if predict_iso == actual_iso:
        print("\nCorrect!")
        print(actual_iso)
        """print("G_with_G.fun")
        print(G_with_G.fun)
        print("G_with_G.status")
        print(G_with_G.status)
        print("G_prime_with_G_prime.fun")
        print(G_prime_with_G_prime.fun)
        print("G_prime_with_G_prime.status")
        print(G_prime_with_G_prime.status)
        print("G_with_G_prime.status")
        print(G_with_G_prime.status)
        print("G_with_G_prime.fun")
        print(G_with_G_prime.fun)
        print("")"""
    else:
        print("Incorrect!")
        print("Actual:")
        print(actual_iso)
        """print("\nG_with_G:")
        print(G_with_G)
        print("\nG_prime_with_G_prime:")
        print(G_prime_with_G_prime)
        print("\nG_with_G_prime:")
        print(G_with_G_prime)"""
        print("G's edges:")
        print(G.edges())
        print("G_prime's edges:")
        print(G_prime.edges())
        break
        #G's edges:
        #[(0, 2), (1, 3), (1, 4), (1, 7), (2, 3), (2, 5), (2, 6), (3, 5), (3, 7), (4, 6), (5, 6)]
        #G_prime's edges:
        #[(0, 3), (1, 5), (1, 7), (2, 4), (2, 6), (3, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]

