use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use petgraph::graph::{DiGraph, NodeIndex};
use rayon::prelude::*;
use nalgebra::{DMatrix, DVector};
use numpy::PyArray1;
use dashmap::DashMap;
use nalgebra::linalg::SymmetricEigen;
use std::collections::HashMap;
use itertools::Itertools;  
use sprs::{CsMat, CsVec, TriMat};
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;

fn build_graph(adjacency_matrix: &CsMat<f64>) -> DiGraph<(), f64> {
    let mut graph = DiGraph::<(), f64>::new();
    let nodes: Vec<_> = (0..adjacency_matrix.rows()).map(|_| graph.add_node(())).collect();

    adjacency_matrix.outer_iterator().enumerate().for_each(|(i, outer)| {
        outer.iter().for_each(|(j, &weight)| {
            if weight != 0.0 {
                graph.add_edge(nodes[i], nodes[j], weight);
            }
        });
    });

    graph
}

fn compute_eigenvector_centrality(graph: &DMatrix<f64>) -> DVector<f64> {
    let eigen = SymmetricEigen::new(graph.clone());
    eigen.eigenvectors.column(eigen.eigenvalues.imax()).into()
}

fn pagerank_power_sprs(adj_matrix: &CsMat<f64>, p: f64, personalize: &CsVec<f64>, tol: f64, max_iter: usize) -> CsVec<f64> {
    let n = adj_matrix.rows();
    let mut pr = CsVec::new(n, (0..n).collect::<Vec<_>>(), vec![1.0 / n as f64; n]);

    for _ in 0..max_iter {
        let old_pr = pr.clone();
        pr = adj_matrix * &old_pr;
        pr = pr.map(|v| v * p) + personalize.map(|v| v * (1.0 - p));

        let sum_pr: f64 = pr.iter().map(|(_, v)| *v).sum();
        pr = pr.map(|v| v / sum_pr);  // Normalize

        let diff: f64 = pr.iter().zip(old_pr.iter()).map(|((_, v1), (_, v2))| (v1 - v2) * (v1 - v2)).sum();
        if diff < tol * tol {
            break;
        }
    }

    pr
}

fn personalized_pagerank(graph: &DiGraph<(), f64>, node_index: usize, alpha: f64, top: usize) -> Vec<usize> {
    let n = graph.node_count();

    // Build the adjacency matrix using sprs
    let mut triplet = TriMat::new((n, n));
    graph.edge_references().for_each(|edge| {
        let source = edge.source().index();
        let target = edge.target().index();
        let weight = *edge.weight();
        triplet.add_triplet(source, target, weight);
    });
    let adj_matrix: CsMat<f64> = triplet.to_csr();

    // Personalization vector
    let personalize = CsVec::new(n, vec![node_index], vec![1.0]);

    // Compute PageRank using power iteration with sparse matrices
    let tol = 1e-6;
    let max_iter = 100;
    let pr = pagerank_power_sprs(&adj_matrix, alpha, &personalize, tol, max_iter);

    // Get top-k indices
    (0..n)
        .filter(|&i| i != node_index)
        .map(|i| (i, pr.get(i).copied().unwrap_or(0.0)))
        .sorted_by(|&(_, pr_a), &(_, pr_b)| pr_b.partial_cmp(&pr_a).unwrap_or(Ordering::Equal))
        .take(top)
        .map(|(i, _)| i)
        .collect()
}

fn get_circle_of_trust_per_node(graph: &DiGraph<(), f64>, alpha: f64, top: usize, _num_cores: usize) -> Vec<Vec<usize>> {
    (0..graph.node_count()).into_par_iter().map(|node_index| {
        personalized_pagerank(&graph, node_index, alpha, top)
    }).collect()
}

fn salsa(graph: &DiGraph<(), f64>, node_index: usize, cot: &[usize], top: usize) -> Vec<usize> {
    let mut bipartite_graph = DiGraph::new();
    let hubs: Vec<_> = cot.iter().map(|&vi| bipartite_graph.add_node(format!("h{}", vi))).collect();
    let mut authorities: HashMap<usize, NodeIndex> = HashMap::new();

    cot.iter().for_each(|&vi| {
        graph.neighbors(NodeIndex::new(vi)).for_each(|neighbor| {
            let vj = neighbor.index();
            let authority_node = authorities.entry(vj).or_insert_with(|| bipartite_graph.add_node(vj.to_string()));
            if let Some(hub_index) = cot.iter().position(|&x| x == vi) {
                bipartite_graph.add_edge(hubs[hub_index], *authority_node, ());
            }
        });
    });

    let n = bipartite_graph.node_count();
    if n == 0 {
        return vec![];
    }

    let mut adj_matrix = DMatrix::zeros(n, n);
    bipartite_graph.edge_indices().for_each(|edge| {
        if let Some((source, target)) = bipartite_graph.edge_endpoints(edge) {
            adj_matrix[(source.index(), target.index())] = 1.0;
            adj_matrix[(target.index(), source.index())] = 1.0;
        }
    });

    let centrality = compute_eigenvector_centrality(&adj_matrix);
    let mut centrality_vec: Vec<_> = centrality.iter().copied().enumerate().collect();
    centrality_vec.sort_unstable_by(|&(_, c1), &(_, c2)| c2.partial_cmp(&c1).unwrap());

    centrality_vec.into_iter()
        .filter(|&(n, _)| n < graph.node_count() && !cot.contains(&n) && n != node_index)
        .take(top)
        .map(|(n, _)| n)
        .collect()
}

fn frequency_by_who_to_follow(graph: &DiGraph<(), f64>, cot_per_node: Vec<Vec<usize>>, _alpha: f64, top: usize, _num_cores: usize) -> Vec<usize> {
    let frequency = DashMap::new();

    cot_per_node.into_par_iter().enumerate().for_each(|(node_index, cot)| {
        let result = salsa(&graph, node_index, &cot, top);
        result.iter().for_each(|&node| {
            *frequency.entry(node).or_insert(0) += 1;
        });
    });

    let mut freq_vec = vec![0; graph.node_count()];
    frequency.into_iter().for_each(|(node, count)| {
        freq_vec[node] = count;
    });

    freq_vec
}

#[pyfunction]
fn wtf_full(data: &PyArray1<f64>, indices: &PyArray1<i32>, indptr: &PyArray1<i32>, rows: usize, cols: usize, topk: usize, num_cores: usize) -> PyResult<Py<PyArray1<usize>>> {
    let data = unsafe { data.as_slice().or_else(|_| Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get data slice")))? };
    let indices = unsafe { indices.as_slice().or_else(|_| Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get indices slice")))? };
    let indptr = unsafe { indptr.as_slice().or_else(|_| Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get indptr slice")))? };

    let indices: Vec<usize> = indices.iter().map(|&x| x as usize).collect();
    let indptr: Vec<usize> = indptr.iter().map(|&x| x as usize).collect();

    let adj_matrix = CsMat::new((rows, cols), indptr, indices, data.to_vec());

    let graph = build_graph(&adj_matrix);

    let cot_per_node = get_circle_of_trust_per_node(&graph, 0.85, topk, num_cores);
    let wtf = frequency_by_who_to_follow(&graph, cot_per_node, 0.85, topk, num_cores);

    let gil = Python::acquire_gil();
    let py = gil.python();
    let wtf_array = PyArray1::from_vec(py, wtf);
    Ok(wtf_array.to_owned())
}

#[pymodule]
fn fast_wtf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(wtf_full, m)?)?;
    Ok(())
}
