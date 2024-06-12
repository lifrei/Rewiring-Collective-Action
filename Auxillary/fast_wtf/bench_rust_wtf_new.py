# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:27:02 2024

@author: Jordan
"""

#%%
import os 
import sys
# Ensure the parent directory is in the sys.path for auxiliary imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import numpy as np
from scipy.sparse import csr_matrix
from fast_wtf import wtf_full
import networkx as nx
from joblib import Parallel, delayed
from fast_pagerank import pagerank_power, pagerank
from scipy.sparse import csr_matrix
from collections import Counter
from netin import DPAH, PATCH, viz, stats
from scipy import sparse
import rustworkx as rx
import time
import timeit

#%%


def nx_sort(G, nodeIndex):

    pp = dict.fromkeys(G.nodes(), 0)
    pp.update({nodeIndex: 1.0})
    print(pp)
    p3 = nx.pagerank(G, personalization = pp)
    
    return p3

def wtf1(G, nodeIndex, topk=5):
        print("old")
        TOPK = topk
        nodes = G
        A = nx.to_scipy_sparse_matrix(G, nodes, format='csr')
        num_cores = os.cpu_count()
        top = TOPK
    
        def _ppr(nodeIndex, A, p, top):
            n = A.shape[0]
            pp = np.zeros(n)
            pp[nodeIndex] = 1
            pr = pagerank(A, p=p, personalize=pp)
            
            pr_indices = np.argpartition(pr, -top-1)[-top-1:]
            pr_indices = pr_indices[np.argsort(pr[pr_indices])[::-1]]
            
            # if nodeIndex == 3:
            #     global pr_2
            #     pr_2 = pr
            
            return pr_indices[pr_indices != nodeIndex][:top]
        
    
        def get_circle_of_trust_per_node(A, p=0.85, top=top, num_cores=num_cores):
            return Parallel(n_jobs=num_cores, prefer="threads")(
                delayed(_ppr)(nodeIndex, A, p, top) for nodeIndex in range(A.shape[0])
            )
    
        def frequency_by_circle_of_trust(A, cot_per_node=None, p=0.85, top=10, num_cores=num_cores):
            if cot_per_node is None:
                cot_per_node = get_circle_of_trust_per_node(A, p, top, num_cores)
            unique_elements, counts_elements = np.unique(np.concatenate(cot_per_node), return_counts=True)
            count_dict = dict(zip(unique_elements, counts_elements))
            return [count_dict.get(nodeIndex, 0) for nodeIndex in range(A.shape[0])]
    
        def _salsa(nodeIndex, cot, A, top=10):
            BG = nx.Graph()
            BG.add_nodes_from(['h{}'.format(vi) for vi in cot], bipartite=0)  # hubs
            edges = [('h{}'.format(vi), int(vj)) for vi in cot for vj in A[vi].indices]
            BG.add_nodes_from(set(e[1] for e in edges), bipartite=1)  # authorities
            BG.add_edges_from(edges)
            centrality = Counter({
                n: c for n, c in nx.eigenvector_centrality_numpy(BG).items()
                if isinstance(n, int) and n not in cot and n != nodeIndex and n not in A[nodeIndex].indices
            })
            return np.array([n for n, _ in centrality.most_common(top)])
    
        def frequency_by_who_to_follow(A, cot_per_node=None, p=0.85, top=top, num_cores=num_cores):
            if cot_per_node is None:
                cot_per_node = get_circle_of_trust_per_node(A, p, top, num_cores)
            results = Parallel(n_jobs=num_cores, prefer="threads")(
                delayed(_salsa)(nodeIndex, cot, A, top) for nodeIndex, cot in enumerate(cot_per_node)
            )
            unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
            count_dict = dict(zip(unique_elements, counts_elements))
            return [count_dict.get(nodeIndex, 0) for nodeIndex in range(A.shape[0])]
    
        def wtf_full_old(A, njobs):
            cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=njobs)
            wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)
            return wtf

        # Calculate recommendations and rewire the graph
        ranking = wtf_full_old(A, njobs= num_cores-1)
        return ranking

#%%
def wtf_rx(G, nodeIndex, topk=5, alpha=0.70, max_iter=50):
    print("new")
    
    TOPK = topk

    # Convert NetworkX graph to rustworkx graph
    start_1 = time.time()
    G = rx.networkx_converter(G)
    end_1 = time.time()
    
    mins = (end_1 - start_1) / 60
    sec = (end_1 - start_1) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')

    def _ppr(nodeIndex, G, p, top, max_iter):
        pp = {node: 0 for node in G.nodes()}
        pp[nodeIndex] = 1.0
        pr = rx.pagerank(G, alpha=p, personalization=pp, max_iter=max_iter)
        pr_values = np.array(list(pr.values()))
        pr_indices = np.argsort(pr_values)[::-1]
        pr_indices = pr_indices[pr_indices != nodeIndex][:top]
        return pr_indices

    def get_circle_of_trust_per_node(G, p, top, max_iter):
        return [_ppr(nodeIndex, G, p, top, max_iter) for nodeIndex in range(len(G.nodes()))]

    def frequency_by_circle_of_trust(G, cot_per_node, top):
        unique_elements, counts_elements = np.unique(np.concatenate(cot_per_node), return_counts=True)
        count_dict = {el: count for el, count in zip(unique_elements, counts_elements)}
        return [count_dict.get(nodeIndex, 0) for nodeIndex in range(len(G.nodes()))]

    def _salsa(nodeIndex, cot, G, top):
        BG = rx.PyGraph()
        hubs = [f'h{vi}' for vi in cot]
        hub_indices = BG.add_nodes_from(hubs)
        edges = [(f'h{vi}', vj) for vi in cot for vj in G.neighbors(vi)]
        authorities = list(set(e[1] for e in edges))
        auth_indices = BG.add_nodes_from(authorities)
        
        hub_index_map = {h: idx for idx, h in enumerate(hubs)}
        auth_index_map = {a: idx for idx, a in enumerate(authorities)}
        
        edges = [(hub_index_map[f'h{vi}'], auth_index_map[vj]) for vi, vj in edges if f'h{vi}' in hub_index_map and vj in auth_index_map]
        
        BG.add_edges_from(edges)
        centrality = rx.eigenvector_centrality(BG)
        centrality = {n: c for n, c in centrality.items() if isinstance(n, int) and n not in cot and n != nodeIndex and n not in G.neighbors(nodeIndex)}
        sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        return np.array([n for n, _ in sorted_centrality[:top]])

    def frequency_by_who_to_follow(G, cot_per_node, top):
        results = [_salsa(nodeIndex, cot, G, top) for nodeIndex, cot in enumerate(cot_per_node)]
        unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
        count_dict = {el: count for el, count in zip(unique_elements, counts_elements)}
        return [count_dict.get(nodeIndex, 0) for nodeIndex in range(len(G.nodes()))]

    def wtf_full_old(G, alpha, top, max_iter):
        cot_per_node = get_circle_of_trust_per_node(G, alpha, top, max_iter)
        wtf = frequency_by_who_to_follow(G, cot_per_node, top)
        return wtf

    # Calculate recommendations and rewire the graph
    ranking = wtf_full_old(G, alpha, TOPK, max_iter)
    return ranking




#G = nx.barabasi_albert_graph(300, 3)
#G = nx.scale_free_graph(600)
G = DPAH(500, f_m=0.5, d=0.1, h_MM=0.5, h_mm=0.5, plo_M=2.0, plo_m=2.0,
         seed = 42)
G.generate()
#G = nx.read_gpickle("DPAH.p")
adjacency_matrix = nx.to_scipy_sparse_array(G, format='csr')
# nx.write_gpickle(G, "DPAH.p")

# Convert scipy.sparse matrix to numpy arrays for data, indices, and indptr
data = np.array(adjacency_matrix.data, dtype=np.float64)
indices = np.array(adjacency_matrix.indices, dtype=np.int32)
indptr = np.array(adjacency_matrix.indptr, dtype=np.int32)

num_cores = os.cpu_count()
start = time.time()
# Call the Rust function

result = wtf_full(data, indices, indptr, adjacency_matrix.shape[0], adjacency_matrix.shape[1], topk=10, num_cores=num_cores-1)
#result = wtf1(G, nodeIndex =3, topk = 10)
#result = wtf_rx(G, nodeIndex = 3, topk = 10)
#pr_3 = nx_sort(G, nodeIndex =3)
#print(result, len(result))
#print(f'{pr_1}\n{pr_2}\n{pr_3}')

end = time.time()
mins = (end - start) / 60
sec = (end - start) % 60
print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')

#%%
# A = np.array([[0,1], [0, 2], [1, 2],[2,0],[3,2]])
# weights = [1,1,1,1,1]
# personalize = np.array([0.4, 0.2, 0.2, 0.4])
# personalize_dic = {i:personalize[i] for i in range(personalize.shape[0])}
# G = sparse.csr_matrix((weights, (A[:,0], A[:,1])), shape=(4, 4))
# pr=pagerank_power(G, p=0.85, personalize = personalize)

# G = nx.from_scipy_sparse_array(G, create_using = nx.DiGraph)
# Gx = rx.networkx_converter(G)
# pr_2 = nx.pagerank(G, personalization = personalize_dic)
# pr_3 = rx.pagerank(Gx, personalization = personalize_dic)

# print(f'{pr}\n{pr_2}\n{pr_3}')



