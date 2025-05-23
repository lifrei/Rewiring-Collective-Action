# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:31:29 2024

@author: Jordan
"""

import os
import pandas as pd 
from itertools import repeat, product
import time
import multiprocessing
import models_checks
import numpy as np 
import glob
from sweep_utils import get_sweep_id, save_sweep_config
from datetime import date


def init(lock_):
    models_checks.init_lock(lock_)
    
    
def get_optimal_process_count():
    total_cpus = multiprocessing.cpu_count()
    
    # Reserve at least 2 cores for system operations
    reserved_cpus = max(2, int(0.25 * total_cpus))
    
    # Use at most 75% of available CPUs
    process_count_opt = max(1, int(0.70 * (total_cpus - reserved_cpus)))
    
    # process_count = int(0.3*total_cpus)
    
    return process_count_opt    

if __name__ == '__main__':
    # Constants and Variables
    numberOfSimulations = 30
    numberOfProcessors = get_optimal_process_count()
    lock = multiprocessing.Lock()
    
    pool = multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,))
    start = time.time()
    
    # Network configuration
    rewiring_list_h = ["diff", "same"]
    directed_topology_list = ["DPAH", "Twitter"]
    undirected_topology_list = ["cl", "FB"]
    
    # Create combined scenarios list
    combined_list1 = [(scenario, rewiring, topology)
                      for scenario in ["biased", "bridge"]
                      for rewiring in rewiring_list_h
                      for topology in directed_topology_list + undirected_topology_list]
    
    combined_list2 = [("node2vec", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list3 = [("None", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list4 = [("wtf", "None", topology) for topology in directed_topology_list]
    # Add combinations for "rand" scenario on all topologies
    combined_list_rand = [("random", "None", topology) for topology in directed_topology_list + undirected_topology_list]

    combined_list = combined_list1 + combined_list_rand + combined_list2 + combined_list3 + combined_list4 

    #combined_list = [("biased", "diff", "cl"), ("bridge", "diff", "cl") ]
    
    # Parameter sweep configuration
    parameter_names = ["politicalClimate"]
    parameters = {
        "politicalClimate": np.linspace(0, 1, 10)
    }
    param_product = [dict(zip(parameters.keys(), x)) for x in product(*parameters.values())]

    # Initialize list to store results
    results = []

    sweep_id = get_sweep_id(parameter_names)

    save_sweep_config(
    sweep_id=sweep_id,
    parameter_names=parameter_names,
    parameters=parameters,
    combined_list=combined_list,
    num_simulations=numberOfSimulations,
    models_checks_module=models_checks
)

    
    # Main sweep loop
    for params in param_product:    
        print(f"Started parameter combination: {params}")
        
        for algo, mode, topology in combined_list:
            print(f"Running scenario: {algo}_{mode}_{topology}")
            
            # Set network size based on topology
            if topology == "Twitter":
                top_file = "twitter_graph_N_789.gpickle"
                nwsize = 789
            elif topology == "FB":
                top_file = "FB_graph_N_786.gpickle"
                nwsize = 786
            else:
                top_file = None
                nwsize = 800

            # Prepare simulation arguments
            sim_args = {
                "rewiringAlgorithm": algo,
                "nwsize": nwsize,
                "rewiringMode": mode,
                "type": topology,
                "top_file": top_file,
                "timesteps": 45000,
                "plot": False,
                **params
            }

            # Run simulations
            sims = pool.starmap(models_checks.simulate, 
                              zip(range(numberOfSimulations), repeat(sim_args)))
            
            for sim in sims:
                results.append({
                    'state': sim.states[-1],  # Store individual final state
                    'state_std': sim.statesds[-1],
                    'political_climate': params['politicalClimate'],
                    'rewiring': mode,
                    'mode': algo,
                    'topology': topology,
                    #'run': sim.run_id  # Add run identifier
                })

    pool.close()
    pool.join()

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Save results
    today = date.today()
    date_str = today.strftime("%b_%d_%Y")
    fname = f'../Output/heatmap_sweep_{sweep_id}.csv'
    results_df.to_csv(fname, index=False)

    # Clean up any temporary files
    for file in glob.glob("*embeddings*"):
        os.remove(file)

    # Print total runtime
    end = time.time()
    total_hours = (end - start) / 3600
    remaining_mins = ((end - start) % 3600) / 60
    print(f'Total runtime: {int(total_hours)} hours, {int(remaining_mins)} minutes')