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


fn compute_eigenvector_centrality(matrix: &DMatrix<f64>) -> DVector<f64> {
    let eigen = SymmetricEigen::new(matrix.clone());
    let max_eigenvalue_index = eigen.eigenvalues.imax();
    eigen.eigenvectors.column(max_eigenvalue_index).into()
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

fn personalized_pagerank(adj_matrix: &CsMat<f64>, node_index: usize, alpha: f64, top: usize) -> Vec<usize> {
    let n = adj_matrix.rows();

    // Personalization vector
    let personalize = CsVec::new(n, vec![node_index], vec![1.0]);

    // Compute PageRank using power iteration with sparse matrices
    let tol = 1e-6;
    let max_iter = 100;
    let pr = pagerank_power_sprs(adj_matrix, alpha, &personalize, tol, max_iter);

    // Get top-k indices
    (0..n)
        .filter(|&i| i != node_index)
        .sorted_by(|&a, &b| {
            let a_val = pr.get(a).copied().unwrap_or(0.0);
            let b_val = pr.get(b).copied().unwrap_or(0.0);
            b_val.partial_cmp(&a_val).unwrap_or(Ordering::Equal)
        })
        .take(top)
        .collect()
}

fn get_circle_of_trust_per_node(adj_matrix: &CsMat<f64>, alpha: f64, top: usize, num_cores: usize) -> Vec<Vec<usize>> {
    (0..adj_matrix.rows()).into_par_iter().map(|node_index| {
        personalized_pagerank(adj_matrix, node_index, alpha, top)
    }).collect()
}


fn salsa(adj_matrix: &CsMat<f64>, node_index: usize, cot: &[usize], top: usize) -> Vec<usize> {
    let mut bipartite_graph = Vec::new();
    let mut authorities: HashMap<usize, usize> = HashMap::new();

    cot.iter().for_each(|&vi| {
        if let Some(view) = adj_matrix.outer_view(vi) {
            view.iter().for_each(|(vj, _)| {
                let authority_node = *authorities.entry(vj).or_insert_with(|| bipartite_graph.len());
                if !bipartite_graph.contains(&(vi, authority_node)) {
                    bipartite_graph.push((vi, authority_node));
                }
            });
        }
    });

    let n = bipartite_graph.len();

    if n == 0 {
        return vec![];
    }

    let mut adj_matrix = DMatrix::zeros(n, n);

    bipartite_graph.iter().for_each(|&(source, target)| {
        if source < n && target < n {
            adj_matrix[(source, target)] = 1.0;
            adj_matrix[(target, source)] = 1.0;
        }
    });

    let centrality = compute_eigenvector_centrality(&adj_matrix);
    let mut centrality_vec: Vec<_> = centrality.iter().copied().enumerate().collect();
    centrality_vec.sort_unstable_by(|&(_, c1), &(_, c2)| c2.partial_cmp(&c1).unwrap());

    centrality_vec.into_iter()
        .filter(|&(n, _)| n < adj_matrix.nrows() && !cot.contains(&n) && n != node_index)
        .take(top)
        .map(|(n, _)| n)
        .collect()
}


fn frequency_by_who_to_follow(adj_matrix: &CsMat<f64>, cot_per_node: Vec<Vec<usize>>, top: usize, num_cores: usize) -> Vec<usize> {
    let results: Vec<Vec<usize>> = cot_per_node.into_par_iter().enumerate().map(|(node_index, cot)| {
        salsa(adj_matrix, node_index, &cot, top)
    }).collect();

    let mut frequency = vec![0; adj_matrix.rows()];

    results.iter().for_each(|result| {
        result.iter().for_each(|&node| {
            if node < frequency.len() {
                frequency[node] += 1;
            }
        });
    });

    frequency
}



#[pyfunction]
fn wtf_full(data: &PyArray1<f64>, indices: &PyArray1<i32>, indptr: &PyArray1<i32>, rows: usize, cols: usize, topk: usize, num_cores: usize) -> PyResult<Py<PyArray1<usize>>> {
    let data = unsafe { data.as_slice().or_else(|_| Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get data slice")))? };
    let indices = unsafe { indices.as_slice().or_else(|_| Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get indices slice")))? };
    let indptr = unsafe { indptr.as_slice().or_else(|_| Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get indptr slice")))? };

    let indices: Vec<usize> = indices.iter().map(|&x| x as usize).collect();
    let indptr: Vec<usize> = indptr.iter().map(|&x| x as usize).collect();

    let adj_matrix = CsMat::new((rows, cols), indptr, indices, data.to_vec());

    let cot_per_node = get_circle_of_trust_per_node(&adj_matrix, 0.85, topk, num_cores);
    let wtf = frequency_by_who_to_follow(&adj_matrix, cot_per_node, topk, num_cores);

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
