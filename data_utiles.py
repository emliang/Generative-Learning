import torch
import numpy as np
import os
from itertools import combinations
import pickle
import copy
import networkx as nx
import gurobipy as gp
from gurobipy import Model, GRB
import scipy
import copy
import time
import random

torch.set_default_dtype(torch.float32)


def Opt_Dataset(args):
    if args['data_set'] == 'max_clique':
        return Max_Clique(args)
    elif args['data_set'] == 'max_is':
        return Max_IS(args)
    elif args['data_set'] == 'max_cut':
        return Max_Cut(args)
    elif args['data_set'] == 'min_cover':
        return Min_Cover(args)
    elif args['data_set'] == 'tsp':
        return TSP(args)
    elif args['data_set'] == 'acopf':
        return ACOPF(args)
    elif args['data_set'] == 'acpf':
        return ACPF(args)


""""""""""""""""""""""""
"Toy Example"
""""""""""""""""""""""""
class Toy_Dataset:
    def __init__(self, args):
        """
        Dataset class to generate test instances.
        param args: Dictionary containingen various arguments and settingens.
        """
        self.x, self.y, self.xt, self.yt = self.generate_data(args['data_type'], args['data_size'])
        self.x_train = torch.as_tensor(self.x).view(-1, 1).to(dtype=torch.float32)
        self.y_train = torch.as_tensor(self.y).view(-1, 1).to(dtype=torch.float32)

    def multi_valued_mapping(self, x, data_type, sample_type=None):
        x = np.reshape(x, [len(x),1])
        if data_type == 1:
            y = [x, -x]
        elif data_type == 2:
            y = [3 * x ** 2 - 0.5, -3 * x ** 2 + 0.5, 
                 x + 1, -x + 1, x - 1, -x - 1]
        elif data_type == 3:
            y = [np.sin(x * np.pi), 2*np.cos(x * np.pi), 
                 -np.sin(x * np.pi), -2*np.cos(x * np.pi),
                 x + 1, -x + 1, x - 1, -x - 1]
        y = np.concatenate(y, axis=1) 
        if sample_type == 'random':
            y = np.array([np.random.choice(y[i]) for i in range(np.shape(y)[0])])
        return y

    def generate_data(self, data_type, data_size):
        x_train = np.linspace(-1,1, data_size)
        y_train = self.multi_valued_mapping(x_train, data_type, 'random')
        x_target = np.linspace(-1, 1, 10000)
        y_target = self.multi_valued_mapping(x_target, data_type)
        return x_train, y_train, x_target, y_target




""""""""""""""""""""""""
"Graph Optimization"
""""""""""""""""""""""""
from networkx.algorithms.approximation import clique, vertex_cover, maxcut
# import igraph as ig

def save_to_single_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_single_pickle(filename):
    with open(filename, 'rb') as file:
        graphs = pickle.load(file)
    return graphs

class Base_Problem:
    def __init__(self, args):
        graph_dim = args['graph_dim']
        graph_sparsity = args['graph_sparsity']
        data_dim = args['data_dim']
        test_dim = args['test_dim']
        # train_dim = data_dim - test_dim
        if not os.path.exists(f'dataset/{self.__class__.__name__}'):
            os.makedirs(f'dataset/{self.__class__.__name__}')
        try:
            graph_data = load_from_single_pickle(
                f'dataset/{self.__class__.__name__}/data_{graph_dim}_{data_dim}.npy')
            print('Load data successfully')
        except:
            print('No available data, create new')
            graph_data = self.generate_data(data_dim, graph_dim, graph_sparsity)
            graph_data = self.opt_solve(graph_data)
            save_to_single_pickle  (graph_data, f'dataset/{self.__class__.__name__}/data_{graph_dim}_{data_dim}.npy')
        if args['network'] == 'mlp':
            node_data, _, solutions = self.get_mlp_data(graph_data, graph_dim)
        else:
            node_data, _, solutions = self.get_graph_data(graph_data, graph_dim)
        x_train = node_data[:-test_dim]
        x_test = node_data[-test_dim:]
        y_train = solutions[:-test_dim]
        y_test = solutions[-test_dim:]
        # graph_train = graph_data[:-test_dim]
        graph_test = graph_data[-test_dim:]

        self.x_train = torch.as_tensor(x_train).to(dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train).to(dtype=torch.float32)
        # self.graph_train = graph_train  # torch.tensor(graph_train ).view(train_dim, graph_dim, graph_dim)
        self.x_test = torch.as_tensor(x_test).to(dtype=torch.float32)
        self.y_test = torch.as_tensor(y_test).to(dtype=torch.float32)
        self.graph_test = graph_test  # torch.tensor(graph_test ).view(test_dim, graph_dim, graph_dim)  

    def generate_data(self, data_dim, graph_dim, sparsity):
        graph_data = []
        for k in range(data_dim):
            n = graph_dim
            # seed = k
            p = np.random.uniform(low=sparsity[0], high=sparsity[1])  # Probability for edge creation
            G = nx.fast_gnp_random_graph(n, p)
            G.remove_edges_from(nx.selfloop_edges(G))
            graph_data.append(G)
        return graph_data

    def get_mlp_data(self, graphs, graph_dim):
        n = len(graphs)
        node_data = np.zeros(shape=[n, (graph_dim * graph_dim - graph_dim) // 2])
        # edge_data = np.zeros(shape=[n, graph_dim, graph_dim])
        solutions = np.zeros(shape=[n, graph_dim])
        for i, G in enumerate(graphs):
            adj_matrix = nx.adjacency_matrix(G).toarray()
            solution = nx.get_node_attributes(G, 'solution')
            # edge_data[i,:,:] = adj_matrix
            solutions[i, :] = np.array([item for item in solution.values()])
            # Get the indices of the lower triangle without the diagonal elements
            upper_triangle_indices = np.triu_indices(graph_dim, k=1)
            # Extract the lower triangle elements using the indices
            upper_triangle_elements = adj_matrix[upper_triangle_indices]
            node_data[i] = upper_triangle_elements
        return node_data, None, solutions

    def get_graph_data(self, graphs, graph_dim):
        n = len(graphs)
        rank = int(graph_dim)
        node_data = np.zeros(shape=[n, graph_dim, rank])
        # edge_data = np.zeros(shape=[n, graph_dim, graph_dim])
        solutions = np.zeros(shape=[n, graph_dim])
        for i, G in enumerate(graphs):
            adj_matrix = nx.adjacency_matrix(G).toarray()
            u, s, vh = np.linalg.svd(adj_matrix)
            s = s ** 0.5
            sq = np.diag(s)
            u = np.matmul(u, sq)
            vh = np.matmul(sq, vh)
            ul = u[:, :rank]
            # vl = vh[:5,:]
            node_data[i] = ul
            solution = nx.get_node_attributes(G, 'solution')
            # edge_data[i,:,:] = adj_matrix
            solutions[i, :] = np.array([item for item in solution.values()])
        return node_data, None, solutions

class Max_Clique(Base_Problem):
    def __init__(self, args):
        super().__init__(args)

    def decoding(self, Gs, node_probabilities):
        clique_list = []
        for i in range(node_probabilities.shape[0]):
            node_prob = node_probabilities[i, :]
            G = Gs[i]
            # Sort nodes by their probabilities in descending order
            # sorted_nodes = sorted(G.nodes, key=lambda node: node_prob[node], reverse=True)
            sorted_nodes = np.argsort(node_prob).tolist()[::-1]
            # Initialize an empty clique
            clique = []
            # Iterate through the sorted nodes
            for node in sorted_nodes:
                if len(clique) == 0:
                    clique.append(node)
                else:
                    # Check if the node is connected to all nodes in the current clique
                    # is_connected_to_clique = all(G[node, clique_node] for clique_node in clique)
                    is_connected_to_clique = all(G.has_edge(node, clique_node) for clique_node in clique)
                    # If the node is connected to all nodes in the clique, add it to the clique
                    if is_connected_to_clique:
                        clique.append(node)
            clique_list.append(len(clique))
        return np.array(clique_list)

    def objective(self, graphs, node_pred):
        return node_pred.sum(-1)

    def violation(self, graphs, node_probabilities):
        vio_list = []
        for i in range(node_probabilities.shape[0]):
            CG = nx.complement(graphs[i])
            node = np.round(node_probabilities[i])
            for k, j in CG.edges():
                if node[k] + node[j] > 1:
                    break
            if node[k] + node[j] > 1:
                vio_list.append(1)
            else:
                vio_list.append(0)
        return np.array(vio_list)

    def opt_solve(self, Gs):
        graph_dim = len(Gs[0].nodes)
        for m, G in enumerate(Gs):
            seed = np.random.randint(int(1e7))
            if graph_dim <= 100:
                ## solve MC for G
                # Create a Gurobi model
                model = gp.Model("MaxClique")
                # Add binary variables for each node
                x = model.addVars(G.nodes, vtype=GRB.BINARY, name="x")
                # Define the objective function
                model.setObjective(gp.quicksum(x[node] for node in G.nodes), GRB.MAXIMIZE)
                # Add constraints: for each pair of non-adjacent nodes (i, j), x_i + x_j <= 1
                CG = nx.complement(G)
                for i, j in CG.edges():
                    model.addConstr(x[i] + x[j] <= 1)
                # Optimize the model
                model.setParam(GRB.Param.OutputFlag, 0)
                model.setParam(GRB.Param.Seed, seed)
                model.optimize()
                vals = model.getAttr('X', x)
                max_clique_nodes = [node for node in G.nodes if vals[node] >= 1]
                max_clique_dim = len(max_clique_nodes)
                if model.Status == GRB.OPTIMAL:
                    print(m, "Maximum clique:", max_clique_dim, end='\r')
                else:
                    print("No optimal solution found.")
            else:
                # iG = ig.Graph(G.edges())
                # all_max_cliques = iG.largest_cliques()
                # max_clique = all_max_cliques[np.random.randint(len(all_max_cliques))]
                # max_clique_dim = len(max_clique)
                max_clique = clique.max_clique(G)
                max_clique_dim = len(max_clique)  
                print(m, "Maximum clique:", max_clique_dim, end='\r')
                vals = {}
                for i in range(graph_dim):
                    if i in max_clique:
                        vals[i] = 1  
                    else:
                        vals[i] = 0
            nx.set_node_attributes(G, vals, 'solution')
        return Gs

class Max_IS(Base_Problem):
    def __init__(self, args):
        super().__init__(args)

    def objective(self, graphs, node_pred):
        return node_pred.sum(-1)

    def decoding(self, Gs, node_probabilities):
        clique_list = []
        for i in range(node_probabilities.shape[0]):
            node_prob = node_probabilities[i, :]
            G = Gs[i]
            # Sort nodes by their probabilities in descending order
            sorted_nodes = np.argsort(node_prob).tolist()[::-1]
            # Initialize an empty clique
            clique = []
            # Iterate through the sorted nodes
            for node in sorted_nodes:
                if len(clique) == 0:
                    clique.append(node)
                else:
                    # Check if the node is connected to all nodes in the current clique
                    # is_connected_to_clique = any(G[node, clique_node] for clique_node in clique)
                    is_connected_to_clique = any(G.has_edge(node, clique_node) for clique_node in clique)
                    # If the node is connected to all nodes in the clique, add it to the clique
                    if not is_connected_to_clique:
                        clique.append(node)
            clique_list.append(len(clique))
        return np.array(clique_list)

    def violation(self, graphs, node_probabilities):
        vio_list = []
        for i in range(node_probabilities.shape[0]):
            G = graphs[i]
            node = np.round(node_probabilities[i])
            for k, j in G.edges():
                if node[k] + node[j] > 1:
                    break
            if node[k] + node[j] > 1:
                vio_list.append(1)
            else:
                vio_list.append(0)
        return np.array(vio_list)

    def opt_solve(self, Gs):
        graph_dim = len(Gs[0].nodes)
        for m, G in enumerate(Gs):
            seed = np.random.randint(int(1e7))
            if graph_dim<=100:
                # Create a Gurobi model
                model = gp.Model("MaxIndependentSet")
                # Add binary variables for each node
                x = model.addVars(G.nodes, vtype=GRB.BINARY, name="x")
                # Define the objective function
                model.setObjective(gp.quicksum(x[node] for node in G.nodes), GRB.MAXIMIZE)
                # Add constraints: for each pair of non-adjacent nodes (i, j), x_i + x_j <= 1
                for i, j in G.edges():
                    model.addConstr(x[i] + x[j] <= 1)
                # Optimize the model
                model.setParam(GRB.Param.OutputFlag, 0)
                model.setParam(GRB.Param.Seed, seed)

                model.optimize()
                vals = model.getAttr('X', x)
                max_IS_nodes = [node for node in G.nodes if vals[node] >= 1]
                # partition = [vals[node] for node in G.nodes]
                # print(self.objective([G], np.array([partition])), model.ObjVal)
                if model.Status == GRB.OPTIMAL:
                    print(m, "Maximum IS:", len(max_IS_nodes), end='\r')
                else:
                    print("No optimal solution found.")
            else:
                max_set = clique.maximum_independent_set(G)
                max_set_dim = len(max_set)
                print(m, "Maximum ind set:", max_set_dim, end='\r')
                vals = {}
                for i in range(graph_dim):
                    if i in max_set:
                        vals[i] = 1  
                    else:
                        vals[i] = 0
            nx.set_node_attributes(G, vals, 'solution')
        return Gs

class Max_Cut(Base_Problem):
    def __init__(self, args):
        super().__init__(args)

    def decoding(self, Gs, node_probabilities, threshold=0.5):
        cut_list = []
        for i in range(node_probabilities.shape[0]):
            x = node_probabilities[i]
            G = Gs[i]
            x[x >= threshold] = 1
            x[x < threshold] = -1
            cut_dim = 0
            for u, v in G.edges():
                cut_dim += (1 - x[u] * x[v]) / 2
            # for u in range(x.shape[0]):
            #     for v in range(x.shape[0]):
            #         if G[u,v]:
            # cut_dim += G[u,v]/2 * (1-x[u]*x[v])
            cut_list.append(cut_dim)
        return np.array(cut_list)

    def objective(self, graphs, node_pred):
        cut_num = []
        for n, G in enumerate(graphs):
            cut_dim = 0
            x = node_pred[n]
            x[x >= 0.5] = 1
            x[x <= 0.5] = -1
            for u, v in G.edges():
                cut_dim += (1 - x[u] * x[v]) / 2
            # for u in range(x.shape[0]):
            #     for v in range(x.shape[0]):
            #         if G[u,v]:
            #             cut_dim += G[u,v]/2 * (1-x[u]*x[v])
            cut_num.append(cut_dim)
        return np.array(cut_num)

    def opt_solve(self, Gs):
        graph_dim = len(Gs[0].nodes)
        for m, G in enumerate(Gs):
            # seed = np.random.randint(int(1e7))
            if graph_dim <= 10:
                # Create a Gurobi model
                model = gp.Model("MaxCut")
                x = model.addVars(G.nodes, vtype=GRB.BINARY, name="x")
                # Create decision variables y_ij for each edge
                y = {}
                for i, j, _ in G.edges(data=True):
                    # w_ij = w_data.get('weight', 1)  # Get the 'weight' attribute or default to 1
                    if j > i:
                        y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
                # Set the objective function
                model.setObjective(
                    gp.quicksum(w_data.get('weight', 1) * y[i, j] for i, j, w_data in G.edges(data=True) if j > i),
                    GRB.MAXIMIZE)
                # Add the linearization constraints
                for i, j, w_data in G.edges(data=True):
                    if j > i:
                        model.addConstr(y[i, j] <= x[i] + x[j])
                        model.addConstr(y[i, j] <= 2- x[j] - x[i])
                # Optimize the model
                model.setParam(GRB.Param.OutputFlag, 0)
                model.setParam(GRB.Param.Seed, seed)
                # Optimize the model
                model.optimize()
                vals = model.getAttr('X', x)
                if np.random.rand() > 0.5:
                    partition = [vals[node]  for node in G.nodes]
                else:
                    partition = [1-vals[node]  for node in G.nodes]
                vals = dict(zip(G.nodes, partition))
                if model.Status == GRB.OPTIMAL:
                    print(m, "Max-Cut Value:", model.ObjVal, end='\r')
                else:
                    print("No optimal solution found.")
            else:  
                cut_size, (partition_1, partition_2) = maxcut.randomized_partitioning(G, p= 0.5)
                cut_size, (partition_1, partition_2) = maxcut.one_exchange(G, partition_1)
                vals = {}
                print(m, "Max-Cut Value:", cut_size, end='\r')
                for node in G.nodes:
                    vals[node] = 1 if node in partition_1 else 0
            nx.set_node_attributes(G, vals, 'solution')
        return Gs

class Min_Cover(Base_Problem):
    def __init__(self, args):
        super().__init__(args)

    def decoding(self, Gs, node_probabilities, threshold=0.5):
        cover_list = []
        for i in range(node_probabilities.shape[0]):
            G = Gs[i]
            vertex_cover = set()
            uncovered_edges = set(G.edges)
            node_prob = node_probabilities[i]
            while uncovered_edges:
                # Calculate greedy scores
                greedy_scores = {u: node_prob[u]  for u in G.nodes if u not in vertex_cover}
                # Select the vertex with the highest greedy score
                selected_vertex = max(greedy_scores, key=greedy_scores.get)
                # Add the selected vertex to the vertex cover
                vertex_cover.add(selected_vertex)
                # Remove all edges incident to the selected vertex
                incident_edges = [(u, v) for u, v in uncovered_edges if u == selected_vertex or v == selected_vertex]
                uncovered_edges -= set(incident_edges)
            cover_list.append(len(vertex_cover))
        return -np.array(cover_list)

    def objective(self, graphs, node_pred):
        cover_num = []
        for n, G in enumerate(graphs):
            vertex_cover = set()
            uncovered_edges = set(G.edges)
            node_prob = node_pred[n]
            while uncovered_edges:
                # Calculate greedy scores
                greedy_scores = {u: node_prob[u]  for u in G.nodes if u not in vertex_cover}
                # Select the vertex with the highest greedy score
                selected_vertex = max(greedy_scores, key=greedy_scores.get)
                # Add the selected vertex to the vertex cover
                vertex_cover.add(selected_vertex)
                # Remove all edges incident to the selected vertex
                incident_edges = [(u, v) for u, v in uncovered_edges if u == selected_vertex or v == selected_vertex]
                uncovered_edges -= set(incident_edges)
            cover_num.append(len(vertex_cover))
        return -np.array(cover_num)

    def opt_solve(self, Gs):
        graph_dim = len(Gs[0].nodes)
        for m, G in enumerate(Gs):
            seed = np.random.randint(int(1e7))
            if graph_dim<=100:
                # Create a Gurobi model
                model = gp.Model("MinCover")
                # Add binary variables for each node
                x = model.addVars(G.nodes, vtype=GRB.BINARY, name="x")
                # Define the objective function
                model.setObjective(gp.quicksum(x[node] for node in G.nodes), GRB.MINIMIZE)
                # Add constraints: for each pair of non-adjacent nodes (i, j), x_i + x_j <= 1
                for i, j in G.edges():
                    model.addConstr(x[i] + x[j] >= 1)
                # Optimize the model
                model.setParam(GRB.Param.OutputFlag, 0)
                model.setParam(GRB.Param.Seed, seed)

                model.optimize()
                vals = model.getAttr('X', x)
                MinCover = [node for node in G.nodes if vals[node] >= 1]
                if model.Status == GRB.OPTIMAL:
                    print(m, "Min COver:", len(MinCover))
                else:
                    print("No optimal solution found.")
            else:
                min_cover = vertex_cover.min_weighted_vertex_cover(G)
                min_cover_dim = len(min_cover)
                print(m, "Minimum ver cover:", min_cover_dim, end='\r')
                vals = {}
                for i in range(graph_dim):
                    if i in min_cover:
                        vals[i] = 1  
                    else:
                        vals[i] = 0 
            nx.set_node_attributes(G, vals, 'solution')
        return Gs

# class TSP(Base_Problem):
#     def __init__(self, args):
#         graph_dim = args['graph_dim']
#         data_dim = args['data_dim']
#         test_dim = args['test_dim']
#         train_dim = data_dim - test_dim
#         if not os.path.exists(f'dataset/{self.__class__.__name__}'):
#             os.makedirs(f'dataset/{self.__class__.__name__}')
#         try:
#             graph_data = load_from_single_pickle(
#                 f'dataset/{self.__class__.__name__}/data_{graph_dim}_{data_dim}.npy')
#             print('Load data successfully')
#         except:
#             print('No available data, create new')
#             graph_data = self.generate_data(data_dim, graph_dim)
#             save_to_single_pickle  (graph_data,
#                                          f'dataset/{self.__class__.__name__}/data_{graph_dim}_{data_dim}.npy')

#         if args['network'] == 'mlp':
#             node_data, _, solutions = self.get_mlp_data(graph_data, graph_dim)
#         else:
#             node_data, _, solutions = self.get_graph_data(graph_data, graph_dim)
#         x_train = node_data[:-test_dim]
#         x_test = node_data[-test_dim:]
#         y_train = solutions[:-test_dim]
#         y_test = solutions[-test_dim:]
#         graph_train = graph_data[:-test_dim]
#         graph_test = graph_data[-test_dim:]

#         self.x_train = torch.tensor(x_train )
#         self.y_train = torch.tensor(y_train )
#         self.graph_train = graph_train
#         self.x_test = torch.tensor(x_test )
#         self.y_test = torch.tensor(y_test )
#         self.graph_test = graph_test

#     def generate_data(self, data_dim, graph_dim):
#         # Callback - use lazy constraints to eliminate sub-tours
#         def subtourelim(model, where):
#             if where == GRB.Callback.MIPSOL:
#                 vals = model.cbGetSolution(model._vars)
#                 # find the shortest cycle in the selected edge list
#                 tour = subtour(vals)
#                 if len(tour) < n:
#                     # add subtour elimination constr. for every pair of cities in tour
#                     model.cbLazy(gp.quicksum(model._vars[i, j]
#                                              for i, j in combinations(tour, 2))
#                                  <= len(tour) - 1)

#         # Given a tuplelist of edges, find the shortest subtour
#         def subtour(vals):
#             # make a list of edges selected in the solution
#             edges = gp.tuplelist((i, j) for i, j in vals.keys()
#                                  if vals[i, j] > 0.5)
#             unvisited = list(range(n))
#             cycle = range(n + 1)  # initial length has 1 more city
#             while unvisited:  # true if list is non-empty
#                 thiscycle = []
#                 neighbors = unvisited
#                 while neighbors:
#                     current = neighbors[0]
#                     thiscycle.append(current)
#                     unvisited.remove(current)
#                     neighbors = [j for i, j in edges.select(current, '*')
#                                  if j in unvisited]
#                 if len(cycle) > len(thiscycle):
#                     cycle = thiscycle
#             return cycle

#         graph_data = []
#         for m in range(data_dim):
#             seed = np.random.randint(int(1e7))
#             n = graph_dim
#             x_coor = {i: np.random.uniform(low=-1, high=1, size=[2]) for i in range(n)}
#             dist = {(i, j): np.sum((x_coor[i] - x_coor[j]) ** 2) ** (0.5) for i in range(n) for j in range(i)}
#             G = nx.complete_graph(n)
#             nx.set_node_attributes(G, x_coor, 'coordinate')
#             nx.set_edge_attributes(G, dist, 'distance')

#             model = gp.Model('TSP')
#             vars = model.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
#             for i, j in vars.keys():
#                 vars[j, i] = vars[i, j]  # edge in opposite direction
#             model.addConstrs(vars.sum(i, '*') == 2 for i in range(n))
#             model._vars = vars

#             model.setParam(GRB.Param.OutputFlag, 0)
#             model.setParam(GRB.Param.Seed, seed)
#             model.Params.LazyConstraints = 1
#             model.optimize(subtourelim)

#             vals = model.getAttr('X', vars)
#             tour = subtour(vals)
#             assert len(tour) == n

#             # print('Optimal tour: %s' % str(tour))
#             print(m, 'Optimal cost: %g' % model.ObjVal, end='\r')
#             nx.set_edge_attributes(G, vals, 'solution')
#             graph_data.append(G)
#         return graph_data

#     def get_mlp_data(self, graphs, graph_dim):
#         n = len(graphs)
#         node_data = np.zeros(shape=[n, graph_dim, 2])
#         edge_data = np.zeros(shape=[n, (graph_dim * graph_dim - graph_dim) // 2])
#         solutions = np.zeros(shape=[n, (graph_dim * graph_dim - graph_dim) // 2])
#         for i, G in enumerate(graphs):
#             coordinate = nx.get_node_attributes(G, 'coordinate')
#             distance = nx.get_edge_attributes(G, 'distance')
#             solution = nx.get_edge_attributes(G, 'solution')
#             node_data[i] = np.array([item for item in coordinate.values()])
#             edge_data[i] = np.array([item for item in distance.values()])
#             solutions[i] = np.array([item for item in solution.values()])
#         return node_data, edge_data, solutions

#     def get_graph_data(self, graphs, graph_dim):
#         n = len(graphs)
#         rank = int(graph_dim ** (0.5)) + 1
#         node_data = np.zeros(shape=[n, graph_dim, 2])
#         edge_data = np.zeros(shape=[n, graph_dim, rank])
#         # solutions = np.zeros(shape=[n, graph_dim, graph_dim])
#         solutions = np.zeros(shape=[n, (graph_dim * graph_dim - graph_dim) // 2])
#         for i, G in enumerate(graphs):
#             # adj_matrix = nx.adjacency_matrix(G).toarray()
#             # u, s, vh = np.linalg.svd(adj_matrix)
#             # s = s ** 0.5
#             # sq = np.diag(s)
#             # u = np.matmul(u, sq)
#             # vh = np.matmul(sq, vh)
#             # ul = u[:, :rank]
#             # edge_data[i] = ul
#             coordinate = nx.get_node_attributes(G, 'coordinate')
#             # distance = nx.get_edge_attributes(G, 'distance')
#             solution = nx.get_edge_attributes(G, 'solution')
#             node_data[i] = np.array([item for item in coordinate.values()])
#             solutions[i] = np.array([item for item in solution.values()])
#             # for (k,v), value in solutions.items():
#             #     solutions[i,k,v] = value
#             #     solutions[i,v,k] = value
#         return node_data, edge_data, solutions

#     def decoding(self, Gs, edge_probability):
#         tour_len_list = []
#         for n, G in enumerate(Gs):
#             graph_dim = len(G.nodes().keys())
#             pred_solution = edge_probability[n]
#             upper_triangle_indices = np.triu_indices(graph_dim, k=1)

#             heat_map = np.ones([graph_dim, graph_dim]) * 0
#             heat_map[upper_triangle_indices] = pred_solution
#             heat_map = heat_map + heat_map.T
#             distance = nx.get_edge_attributes(G, 'distance')
#             tour_len = 0

#             left_node = np.arange(graph_dim).tolist()
#             init_star_node = np.random.randint(graph_dim)
#             tour_seq = [init_star_node]
#             star_node = copy.copy(init_star_node)
#             while len(left_node) > 0:
#                 heat_map[star_node, star_node] = -1e5
#                 next_node = np.argmax(heat_map[star_node, :])
#                 # print(star_node, next_node, left_node, heat_map[star_node, :])
#                 try:
#                     tour_len += distance[(star_node, next_node)]
#                 except:
#                     tour_len += distance[(next_node, star_node)]
#                 tour_seq.append(next_node)
#                 heat_map[:, next_node] = -1e5
#                 heat_map[next_node, star_node] = -1e5
#                 left_node.remove(next_node)
#                 star_node = next_node
#                 tour_seq.append(next_node)
#             tour_len_list.append(tour_len)
#         return np.array(tour_len_list)

#     def objective(self, graphs, node_pred):
#         tour_length = []
#         for n, G in enumerate(graphs):
#             distance = nx.get_edge_attributes(G, 'distance')
#             distance = np.array([item for item in distance.values()])
#             solution = node_pred[n]
#             tour_length.append(np.sum(distance * solution))
#         return tour_length  



""""""""""""""""""""""""
"Power Flow Optimization"
""""""""""""""""""""""""

from pypower.api import opf, makeYbus, runpf, rundcopf, makeBdc
from pypower import idx_bus, idx_gen, idx_brch, ppoption
from pypower.idx_cost import COST

class ACOPF:
    def __init__(self, args):
        nbus = args['graph_dim']
        data_dim = args['data_dim']
        test_dim = args['test_dim']
        train_dim = data_dim - test_dim
        if not os.path.exists(f'dataset/{self.__class__.__name__}'):
            os.makedirs(f'dataset/{self.__class__.__name__}')
        self.load_ppc(nbus)
        X, Y = self.load_data()
        X = X[:data_dim]
        Y = Y[:data_dim]

        self.x_train = X[:-test_dim]
        self.y_train = Y[:-test_dim]
        self.x_test = X[-test_dim:]
        self.y_test = Y[-test_dim:]

    def load_ppc(self, nbus):
        data_path = 'dataset/ACOPF/data'
        mat_data = scipy.io.loadmat(data_path + '/PF/train_case' + str(nbus) + '.mat')
        ppc = mat_data['mpc']
        gen = ppc['gen'][0, 0]
        bus = ppc['bus'][0, 0]
        branch = ppc['branch'][0, 0]
        gencost = ppc['gencost'][0, 0]
        genbase = gen[:, idx_gen.MBASE]
        baseMVA = ppc['baseMVA'][0, 0][0]

        self.ppc = copy.deepcopy({'gen': gen, 'bus': bus, 'branch': branch, 'gencost': gencost, 'genbase': genbase, 'baseMVA': baseMVA})
        
        self.quad_costs = torch.tensor(gencost[:, 4])  
        self.lin_costs = torch.tensor(gencost[:, 5])  
        # initial values for solver
        self.vm_init = torch.tensor(bus[:, idx_bus.VM])  
        self.va_init = torch.tensor(np.deg2rad(bus[:, idx_bus.VA]))  
        self.pg_init = torch.tensor(gen[:, idx_gen.PG] / genbase)  
        self.qg_init = torch.tensor(gen[:, idx_gen.QG] / genbase)  

        self.nbus = bus.shape[0]
        self.ngen = gen.shape[0]

        slack = np.where(bus[:, idx_bus.BUS_TYPE] == 3)[0]
        pv = np.where(bus[:, idx_bus.BUS_TYPE] == 2)[0]
        self.gen_index = np.concatenate([slack, pv])
        self.slack_gen_index = slack
        self.non_slack_gen_index = pv
        self.bus_index = np.arange(self.nbus)
        self.gen_index.sort()
        self.slack_gen_index.sort()
        self.non_slack_gen_index.sort()

        self.non_gen_index = set(self.bus_index).difference(set(self.gen_index))
        self.non_gen_index = np.array(list(self.non_gen_index))

        self.slack_gen_index_ = np.array([np.where(x == self.gen_index)[0][0] for x in self.slack_gen_index])
        self.non_slack_gen_index_ = np.array([np.where(x == self.gen_index)[0][0] for x in self.non_slack_gen_index])

        bus[:, 0] -= 1
        branch[:, 0] -= 1
        branch[:, 1] -= 1
        Ybus, _, _ = makeYbus(baseMVA, bus, branch)
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus))  
        self.Ybusi = torch.tensor(np.imag(Ybus))  
        self.baseMVA = torch.tensor(baseMVA)  
        self.genbase = torch.tensor(genbase)  

        self.Pd = torch.tensor(mat_data['RPd_train'].T)   / self.baseMVA
        self.Qd = torch.tensor(mat_data['RQd_train'].T)   / self.baseMVA
        self.Pg = torch.tensor(mat_data['RPg_train'].T)   / self.genbase
        self.Qg = torch.tensor(mat_data['RQg_train'].T)   / self.genbase
        self.Vm = torch.tensor(mat_data['RVm_train'].T)  
        self.Va = torch.deg2rad(torch.tensor(mat_data['RVa_train'].T)  )

        self.pmax = torch.tensor(gen[:, idx_gen.PMAX])   / self.genbase
        self.pmin = torch.tensor(gen[:, idx_gen.PMIN])   / self.genbase
        self.qmax = torch.tensor(gen[:, idx_gen.QMAX])   / self.genbase
        self.qmin = torch.tensor(gen[:, idx_gen.QMIN])   / self.genbase
        self.vmax = torch.tensor(bus[:, idx_bus.VMAX])  
        self.vmin = torch.tensor(bus[:, idx_bus.VMIN])  
        self.amax = torch.tensor([3.14/2])  
        self.amin = torch.tensor([-3.14/2])  
        self.slack_va = self.va_init[self.slack_gen_index]
        self.u_max = torch.cat([self.pmax, self.qmax, self.vmax, self.va_init+np.pi], dim=0)
        self.u_min = torch.cat([self.pmin, self.qmin, self.vmin, self.va_init-np.pi], dim=0)

        ### pg, qg, vm, va,
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ngen
        self.vm_start_yidx = self.ngen * 2
        self.va_start_yidx = self.ngen * 2 + self.nbus
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus

    def load_data(self):
        X = torch.cat([self.Pd, self.Qd], dim=1)  
        Y = torch.cat([self.Vm, self.Va], dim=1)
        # Y = torch.cat([self.Vm*torch.cos(self.Va), 
        #                self.Vm*torch.sin(self.Va)], dim=1)    
        # Y = torch.cat([self.Pg[:, self.non_slack_gen_index_], self.Vm[:, self.gen_index]], dim=1)  
        # Y = torch.cat([self.Pg, self.Qg, self.Vm, self.Va], dim=1)  
        self.X_mean = torch.mean(X, dim=0, keepdim=True)
        self.X_std = torch.std(X, dim=0, keepdim=True)
        self.Y_mean = torch.mean(Y, dim=0, keepdim=True)
        self.Y_std = torch.std(Y, dim=0, keepdim=True)
        return (X - self.X_mean) / (self.X_std + 1e-8), (Y - self.Y_mean) / (self.Y_std + 1e-8)

    def get_yvars(self, Y):
        pg = Y[:, :self.ngen]
        qg = Y[:, self.ngen:2 * self.ngen]
        vm = Y[:, -2 * self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        # pg, qg, vr, vi = self.get_yvars(Y)
        ## power balance equations
        tmp1 = vr @ self.Ybusr - vi @ self.Ybusi
        tmp2 = -vr @ self.Ybusi - vi @ self.Ybusr
        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=X.device)
        pg_expand[:, self.gen_index] = pg
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr * tmp1 - vi * tmp2)
        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=X.device)
        qg_expand[:, self.gen_index] = qg
        react_resid = (qg_expand - X[:, self.nbus:2*self.nbus]) - (vr * tmp2 + vi * tmp1)
        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)
        return resids
    
    def cur_eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=X.device)
        pg_expand[:, self.gen_index] = pg
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=X.device)
        qg_expand[:, self.gen_index] = qg
        pg = pg_expand - X[:, :self.nbus]
        qg = qg_expand - X[:, self.nbus:2*self.nbus]
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        I_real_calc = (pg*vr + qg*vi) / (vm**2)
        I_imag_calc = (pg*vi - qg*vr) / (vm**2)
        I_real_inj = vr @ self.Ybusr - vi @ self.Ybusi
        I_imag_inj = -vr @ self.Ybusi - vi @ self.Ybusr
        resids = torch.cat([
            I_real_calc -I_imag_calc ,
            I_real_inj -  I_imag_inj
        ], dim=1)
        return resids

    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)
        ineq = torch.cat([
        pg - self.pmax,
        self.pmin - pg,
        qg - self.qmax,
        self.qmin - qg,
        vm - self.vmax,
        self.vmin - vm], dim=1)     
        return torch.clamp(ineq, 0)

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)
        # helper functions
        mdiag = lambda v1, v2: torch.diag_embed(
            torch.multiply(v1, v2))  # torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        Ydiagv = lambda Y, v: torch.multiply(Y.unsqueeze(0), v.unsqueeze(
            1))  # Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        dtm = lambda v, M: torch.multiply(v.unsqueeze(2), M)  # torch.diag_embed(v).bmm(M)

        # helper quantities
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        Yr = self.Ybusr
        Yi = self.Ybusi
        YrvrYivi = vr @ Yr - vi @ Yi
        YivrYrvi = vr @ Yi + vi @ Yr
        # print(cosva.shape, YrvrYivi.shape, Yi.shape, Ydiagv(Yr, -vi).shape)
        # print(1/0)
        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ngen, device=Y.device)
        dreal_dpg[self.gen_index, :] = torch.eye(self.ngen, device=Y.device)
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva) - Ydiagv(Yi, sinva)) \
                    - mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva) + Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi) - Ydiagv(Yi, vr)) \
                    - mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi) + Ydiagv(Yr, vr))

        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ngen, device=Y.device)
        dreact_dqg[self.gen_index, :] = torch.eye(self.ngen, device=Y.device)
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva) + Ydiagv(Yr, sinva)) \
                     - mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva) - Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi) + Ydiagv(Yr, vr)) \
                     - mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi) - Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape),
                       torch.zeros(vr.shape[0], self.nbus, self.ngen, device=Y.device),
                       dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ngen, device=Y.device),
                       dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                       dreact_dvm, dreact_dva], dim=2)], dim=1)
        return jac

    def power_flow_v(self, X, Y):
        ### vm,va -> pg,qg
        vm = Y[:, :self.nbus]
        va = Y[:, self.nbus:self.nbus * 2]
        pd = X[:, :self.nbus]
        qd = X[:, self.nbus:self.nbus * 2]
        # vm = torch.max(vm, self.vmin)
        # vm = torch.min(vm, self.vmax)
        ### power flow
        # vr = Y[:, :self.nbus]
        # vi = Y[:, self.nbus:self.nbus * 2]
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        ## power balance equations
        tmp1 = vr @ self.Ybusr - vi @ self.Ybusi
        tmp2 = -vr @ self.Ybusi - vi @ self.Ybusr
        # real power
        pg_expand = (vr * tmp1 - vi * tmp2) + pd
        # reactive power
        qg_expand = (vr * tmp2 + vi * tmp1) + qd

        pg = pg_expand[:, self.gen_index]
        qg = qg_expand[:, self.gen_index]

        # pg_res = pg_expand[:, self.non_gen_index]
        # qg_res = qg_expand[:, self.non_gen_index]
        return torch.cat([pg, qg, vm, va], dim=1)#, pg_res, qg_res
        # return torch.cat([pg, qg, vr, vi], dim=1)#, pg_res, qg_res

    def power_flow_p(self, X, Yt, tol=1e-5, bsz=1024, max_iters=10):
        ## Step 1: Newton's method
        Y = torch.zeros(X.shape[0], self.nbus * 2 + self.ngen * 2, device=X.device)
        # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
        Y[:, self.pg_start_yidx + self.non_slack_gen_index_] = Yt[:, :len(self.non_slack_gen_index_)]  # pg at non-slack gens
        Y[:, self.vm_start_yidx + self.gen_index] = Yt[:, -self.ngen:]  # vm at gens
        # Y[:, self.pg_start_yidx: self.pg_start_yidx + self.ngen] = Yt[:, : self.ngen]  # pg at all gen
        # Y[:, self.vm_start_yidx: self.vm_start_yidx + self.nbus] = Yt[:, self.ngen: self.ngen + self.nbus]  # vm at all bus
        # Y[:, self.va_start_yidx: self.va_start_yidx + self.nbus] = Yt[:, self.ngen+self.nbus: self.ngen + 2*self.nbus]  # va at all bus


        ### Step 2: clamp mix-max
        gen = Y[:, self.pg_start_yidx: self.pg_start_yidx+self.ngen].clone()
        gen = torch.max(gen, self.pmin)  # element-wise max with broadcasting
        gen = torch.min(gen, self.pmax)  # element-wise min with broadcasting
        Y[:, self.pg_start_yidx: self.pg_start_yidx + self.ngen] = gen

        vm = Y[:, self.vm_start_yidx: self.vm_start_yidx+self.nbus].clone()
        vm = torch.max(vm, self.vmin)  # element-wise max with broadcasting
        vm = torch.min(vm, self.vmax)  # element-wise min with broadcasting
        Y[:, self.vm_start_yidx: self.vm_start_yidx+self.nbus] = vm


        ### Step 3: init guesses for remainingen values
        Y[:, self.vm_start_yidx + self.non_gen_index] = self.vm_init[self.non_gen_index]  # vm at load buses
        Y[:, self.va_start_yidx + self.bus_index] = self.va_init
        Y[:, self.qg_start_yidx: self.qg_start_yidx+self.ngen] = 0  # qg at gens (not used in Newton upd)
        Y[:, self.pg_start_yidx + self.slack_gen_index_] = 0  # pg at slack (not used in Newton upd)

        keep_constr = np.concatenate([
            self.pflow_start_eqidx + self.non_slack_gen_index,  # real power flow at non-slack gens
            self.pflow_start_eqidx + self.non_gen_index,  # real power flow at load buses
            self.qflow_start_eqidx + self.non_gen_index])  # reactive power flow at load buses
        newton_guess_inds = np.concatenate([
            self.vm_start_yidx + self.non_gen_index,  # vm at load buses
            self.va_start_yidx + self.non_slack_gen_index,  # va at non-slack gens
            self.va_start_yidx + self.non_gen_index])  # va at load buses
        converged = torch.zeros(X.shape[0])
        for b in range(0, X.shape[0], bsz):
            X_b = X[b:b + bsz]
            Y_b = Y[b:b + bsz]
            for _ in range(max_iters):
                gy = self.eq_resid(X_b, Y_b)[:, keep_constr]
                jac_full = self.eq_jac(Y_b)
                jac = jac_full[:, keep_constr, :]
                jac = jac[:, :, newton_guess_inds]
                delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
                Y_b[:, newton_guess_inds] -= delta
                if torch.abs(delta).max() < tol:
                    break
            converged[b:b + bsz] = (delta.abs() < tol).all(dim=1)
        ## Step 2: Solve for remainingen variables
        pf_res = -self.eq_resid(X, Y)
        # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
        Y[:, self.qg_start_yidx: self.qg_start_yidx + self.ngen] = pf_res[:, self.qflow_start_eqidx + self.gen_index]
        # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
        Y[:, self.pg_start_yidx + self.slack_gen_index_] = pf_res[:, self.pflow_start_eqidx + self.slack_gen_index]
        return Y

    def scaling_v(self, Y):
        # vm = Y[:, :self.nbus]
        # va = Y[:, self.nbus:self.nbus * 2]
        # vm_scale = vm * (self.vmax - self.vmin) + self.vmin
        # va_scale = va * (self.amax - self.amin) + self.amin
        # return torch.cat([vm_scale, va_scale], dim=1)
        return Y * self.Y_std + self.Y_mean

    def scaling_load(self, X):
        return X * self.X_std + self.X_mean

    def constraint_vio(self, pg, qg, vm):
        ineq = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm], dim=1)
        return torch.clamp(ineq, 0)

    def decoding(self, X, Y):
        # X = torch.tensor(X, device=self.X_mean.device)  
        # Y = torch.tensor(Y, device=self.X_mean.device)  
        X = X * self.X_std + self.X_mean
        Y = Y * self.Y_std + self.Y_mean
        # pg, qg, vm, va = self.power_flow_v(X, Y)
        # pg, qg, vm, va = self.power_flow_p(X, Y)
        pg, qg, vm, va = self.power_flow_all(Y)
        # pg = Y
        # Y = self.opt_proj(X, torch.cat([pg,qg,vm,va],dim=1))
        # pg, qg, vm, va = self.get_yvars(Y)
        ineq_vio = self.ineq_resid(pg, qg, vm)
        eq_vio = self.eq_resid(X, torch.cat([pg,qg,vm,va],dim=1))
        vio = torch.cat([eq_vio, ineq_vio], dim=1).abs()
        vio_rate = torch.sign(vio)/vio.shape[1]
        pg_mw = pg * self.genbase
        cost = (self.quad_costs * pg_mw ** 2).sum(axis=1) + \
               (self.lin_costs * pg_mw).sum(axis=1)
        return cost / (self.genbase.mean() ** 2), vio.sum(1)

    def objective(self, pg):
        pg_mw = pg * self.genbase
        cost = (self.quad_costs * pg_mw ** 2).sum(axis=1) + \
               (self.lin_costs * pg_mw).sum(axis=1)
        return cost / (self.genbase.mean() ** 2)

    def opt_solve(self, X, tol=1e-5):
        X = X * self.X_std + self.X_mean
        X_np = X.detach().cpu().numpy()
        ppc = self.ppc
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
        Y = []
        total_time = 0
        for i in range(X_np.shape[0]):
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA.cpu().numpy()
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA.cpu().numpy()
            start_time = time.time()
            my_result = opf(ppc, ppopt)
            end_time = time.time()
            total_time += (end_time - start_time)
            print(i, end='\r')
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase.cpu().numpy()
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase.cpu().numpy()
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))
        return np.array(Y), total_time / len(X_np)

    def opt_proj(self, X, Y, tol=1e-5):
        pg, qg, vm, va = self.get_yvars(Y)
        pg_all = (pg * self.genbase).detach().cpu().numpy() 
        qg_all = (qg * self.genbase).detach().cpu().numpy()
        vm_all = vm.detach().cpu().numpy()
        va_all = np.rad2deg(va.detach().cpu().numpy())
        X_np = (X * self.baseMVA).detach().cpu().numpy()
        Y = []
        total_time = 0
        start_time = time.time()
        for i in range(X_np.shape[0]):
            print(i, end='\r')
            pg_0 = pg_all[i]
            pd = X_np[i]
            ppc = copy.deepcopy(self.ppc)
            ppc['gencost'][:, COST] = 1
            ppc['gencost'][:, COST + 1] = -2 * pg_0
            # Set reduced voltage bounds if applicable
            ppc['bus'][:, idx_bus.VM] = vm_all[i]
            ppc['bus'][:, idx_bus.VA] = va_all[i]
            ppc['gen'][:, idx_gen.PG] = pg_all[i]
            ppc['gen'][:, idx_gen.QG] = qg_all[i]
            # Solver 1
            ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
            ppc['bus'][:, idx_bus.PD] = pd[:self.nbus] 
            ppc['bus'][:, idx_bus.QD] = pd[self.nbus:] 
            my_result = opf(ppc, ppopt)
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase.cpu().numpy()
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase.cpu().numpy()
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))
        end_time = time.time()
        total_time += (end_time - start_time)
        return torch.tensor(np.array(Y) , device=X.device)

    def opt_pf(self, X, Y=None, tol=1e-5):
        X_np = (X * self.baseMVA).detach().cpu().numpy() 
        Y = []
        total_time = 0
        start_time = time.time()
        for i in range(X_np.shape[0]):
            print(i, end='\r')
            pd = X_np[i]
            ppc = copy.deepcopy(self.ppc)
            # ppc['bus'][:, idx_bus.VM] *= np.random.uniform(0,1)
            # ppc['bus'][:, idx_bus.VA] *= np.random.uniform(0,1)
            # ppc['gen'][:, idx_gen.PG] *= np.random.uniform(0,1)
            # ppc['gen'][:, idx_gen.QG] *= np.random.uniform(0,1)
            ppc['bus'][:, idx_bus.PD] = pd[:self.nbus] 
            ppc['bus'][:, idx_bus.QD] = pd[self.nbus:2*self.nbus] 
            ppopt = ppoption.ppoption(PF_ALG=1, VERBOSE=0, PF_MAX_IT=1000, fname=0)  # MIPS PDIPM
            my_result, success = runpf(ppc, ppopt)
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase.cpu().numpy()
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase.cpu().numpy()
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))
        end_time = time.time()
        total_time += (end_time - start_time)
        return torch.tensor(np.array(Y) , device=X.device), total_time / len(X_np)

    def opt_pf_warmstart(self, X, Y, tol=1e-5):
        pg, qg, vm, va = self.get_yvars(Y)
        pg_all = (pg * self.genbase).detach().cpu().numpy() 
        qg_all = (qg * self.genbase).detach().cpu().numpy()
        vm_all = vm.detach().cpu().numpy()
        va_all = np.rad2deg(va.detach().cpu().numpy())
        X_np = (X * self.baseMVA).detach().cpu().numpy()
        Y = []
        total_time = 0
        start_time = time.time()
        for i in range(X_np.shape[0]):
            print(i, end='\r')
            pg_0 = pg_all[i]
            pd = X_np[i]
            ppc = copy.deepcopy(self.ppc)
            ppc['gencost'][:, COST] = 1
            ppc['gencost'][:, COST + 1] = -2 * pg_0
            # Set reduced voltage bounds if applicable
            ppc['bus'][:, idx_bus.VM] = vm_all[i]
            ppc['bus'][:, idx_bus.VA] = va_all[i]
            # ppc['gen'][:, idx_gen.PG] = pg_all[i]
            # ppc['gen'][:, idx_gen.QG] = qg_all[i]
            # Solver 1
            ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol, PDIPM_MAX_IT=100)  # MIPS PDIPM
            ppc['bus'][:, idx_bus.PD] = pd[:self.nbus] 
            ppc['bus'][:, idx_bus.QD] = pd[self.nbus:] 
            my_result, _ = runpf(ppc, ppopt)
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase.cpu().numpy()
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase.cpu().numpy()
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))
        end_time = time.time()
        total_time += (end_time - start_time)
        return torch.tensor(np.array(Y) , device=X.device), total_time / len(X_np)

class ACPF(ACOPF):
    def __init__(self, args):
        super().__init__(args)

    def decoding(self, X, Y):
        # X = torch.tensor(X, device=self.vmax.device)
        # Y = torch.tensor(Y, device=self.vmax.device)
        # X = self.scaling_load(X)
        # Y = self.scaling_v(Y)
        X_scale = X * self.X_std + self.X_mean
        Y_scale = Y * self.Y_std + self.Y_mean
        Y_full = self.power_flow_v(X_scale, Y_scale)
        eq_vio = self.eq_resid(X_scale, Y_full)
        return torch.abs(eq_vio).mean(1)

    # def power_flow_p(self, X, Yt, tol=1e-5, bsz=1024, max_iters=5):
    #     ## Step 1: Newton's method
    #     Y = Yt
    #     # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
    #     # Y[:, self.pg_start_yidx + self.non_slack_gen_index_] = Yt[:, :len(self.non_slack_gen_index_)]  # pg at non-slack gens
    #     # Y[:, self.vm_start_yidx + self.gen_index] = Yt[:, -self.ngen:]  # vm at gens
    #     # Y[:, self.pg_start_yidx: self.pg_start_yidx + self.ngen] = Yt[:, : self.ngen]  # pg at all gen
    #     # Y[:, self.vm_start_yidx: self.vm_start_yidx + self.nbus] = Yt[:, self.ngen: self.ngen + self.nbus]  # vm at all bus
    #     # Y[:, self.va_start_yidx: self.va_start_yidx + self.nbus] = Yt[:, self.ngen+self.nbus: self.ngen + 2*self.nbus]  # va at all bus


    #     ### Step 2: clamp mix-max
    #     gen = Y[:, self.pg_start_yidx: self.pg_start_yidx+self.ngen].clone()
    #     gen = torch.max(gen, self.pmin)  # element-wise max with broadcasting
    #     gen = torch.min(gen, self.pmax)  # element-wise min with broadcasting
    #     Y[:, self.pg_start_yidx: self.pg_start_yidx + self.ngen] = gen

    #     vm = Y[:, self.vm_start_yidx: self.vm_start_yidx+self.nbus].clone()
    #     vm = torch.max(vm, self.vmin)  # element-wise max with broadcasting
    #     vm = torch.min(vm, self.vmax)  # element-wise min with broadcasting
    #     Y[:, self.vm_start_yidx: self.vm_start_yidx+self.nbus] = vm


    #     ### Step 3: init guesses for remainingen values
    #     Y[:, self.vm_start_yidx + self.non_gen_index] = self.vm_init[self.non_gen_index]  # vm at load buses
    #     Y[:, self.va_start_yidx + self.bus_index] = self.va_init
    #     Y[:, self.qg_start_yidx: self.qg_start_yidx+self.ngen] = 0  # qg at gens (not used in Newton upd)
    #     Y[:, self.pg_start_yidx + self.slack_gen_index_] = 0  # pg at slack (not used in Newton upd)

    #     keep_constr = np.concatenate([
    #         self.pflow_start_eqidx + self.non_slack_gen_index,  # real power flow at non-slack gens
    #         self.pflow_start_eqidx + self.non_gen_index,  # real power flow at load buses
    #         self.qflow_start_eqidx + self.non_gen_index])  # reactive power flow at load buses
    #     newton_guess_inds = np.concatenate([
    #         self.vm_start_yidx + self.non_gen_index,  # vm at load buses
    #         self.va_start_yidx + self.non_slack_gen_index,  # va at non-slack gens
    #         self.va_start_yidx + self.non_gen_index])  # va at load buses
    #     converged = torch.zeros(X.shape[0])
    #     for b in range(0, X.shape[0], bsz):
    #         X_b = X[b:b + bsz]
    #         Y_b = Y[b:b + bsz]
    #         for _ in range(max_iters):
    #             gy = self.eq_resid(X_b, Y_b)[:, keep_constr]
    #             jac_full = self.eq_jac(Y_b)
    #             jac = jac_full[:, keep_constr, :]
    #             jac = jac[:, :, newton_guess_inds]
    #             delta = torch.linalg.solve(jac, gy.unsqueeze(-1)).squeeze(-1)
    #             Y_b[:, newton_guess_inds] -= delta
    #             if torch.abs(delta).max() < tol:
    #                 break
    #         converged[b:b + bsz] = (delta.abs() < tol).all(dim=1)
    #     ## Step 2: Solve for remainingen variables
    #     pf_res = -self.eq_resid(X, Y)
    #     # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
    #     Y[:, self.qg_start_yidx: self.qg_start_yidx + self.ngen] = pf_res[:, self.qflow_start_eqidx + self.gen_index]
    #     # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
    #     Y[:, self.pg_start_yidx + self.slack_gen_index_] = pf_res[:, self.pflow_start_eqidx + self.slack_gen_index]
    #     return self.get_yvars(Y)






""""""""""""""""""""""""
"Inverse Kinematics Problem"
""""""""""""""""""""""""
from scipy.optimize import minimize
class Inv_Kinematics:
    def __init__(self, dof=2):
        self.dof = dof
        self.armlen = [1] * dof
        if not os.path.exists(f'dataset/{self.__class__.__name__}'):
            os.makedirs(f'dataset/{self.__class__.__name__}')
        try:
            [x,y] = load_from_single_pickle(
                f'dataset/{self.__class__.__name__}/data_{dof}.npy')
            print('Load data successfully')
        except:
            print('No available data, create new')
            x, y = self.generate_data()
            dataset = [x,y]
            save_to_single_pickle  (dataset, f'dataset/{self.__class__.__name__}/data_{dof}.npy')

    def generate_data(self):
        # Function to compute forward kinematics
        target_position = np.random.uniform(0, self.dof, size=[1000, 2])
        x = []
        y = []
        for target_point in target_position:
            optimal_thetas = self.opt_solve(target_point)
            solve_point = self.forward_kinematics(self.armlen, optimal_thetas)
            print(np.sum((solve_point-target_point)**2))
            if np.sum((solve_point-target_point)**2)<1e-5:
                x.append(target_point)
                y.append(optimal_thetas)
                print("Optimal angles: ", optimal_thetas)
        return np.array(x), np.array(y)

    def opt_solve(self, target_point):
        # Function to compute the distance between the end of the arm and the target point
        def error(thetas, armlen, target):
            x, y = self.forward_kinematics(thetas, armlen)
            return (x-target[0])**2 + (y-target[1])**2 #+ 0.01*sum(thetas**2)
        initial_thetas = np.random.randn(self.dof)
        # Use scipy's minimize function to minimize the error function
        result = minimize(error, initial_thetas, args=(self.armlen, target_point), method='BFGS', 
                          options={'maxiter': 10000, 'maxfev': 10000})
        # The optimal angles are in result.x
        optimal_thetas = result.x    
        return optimal_thetas 

    @ staticmethod
    def forward_kinematics(thetas, armlen):
        x=0
        y=0
        theta = 0
        for al, th in zip(armlen, thetas):
            theta+= th
            x += al * np.cos(theta)
            y += al * np.sin(theta)
        return np.array([x, y])
    
    
# Inv_Kinematics(2)

