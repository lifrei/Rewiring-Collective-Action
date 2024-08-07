# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:27:39 2022

@author: everall
"""

# ilona presenting first week of november
# 3 November
# %%
import sys
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import stdev, median, mean
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os as os
import matplotlib.ticker as ticker
import inspect
import itertools
import math
import re

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)


# %% import data and joining together

#sys.path.append('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/paper/github_paper')
#os.chdir('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/paper/github_paper')
#rint(os.getcwd())

#change params here
file_extension = "csv"
scenario =  "random" #bridge" #"bias"
parameter =    "stubbornness"#"politicalClimate" #"stubbornness" #

conds = [file_extension, scenario, parameter]

# stipulating regex pattern to get the parameter value from the file name string
def use_regex(input_text):
    pattern = re.compile(r"\d\.\d\d", re.IGNORECASE)

    return pattern.search(input_text)


file_list = os.listdir("../Output")

# get list of relevant output files
file_list = list(filter(
    lambda x: all(cond in x for cond in conds) and "heatmap" not in x, os.listdir("../Output")))

# otherwise define it manually
# file list =

joined = []

for i in file_list:
    abs_path = os.path.join(currentdir, i)

    try:
        value = use_regex(i).group(0)
        print(value)
    except(AttributeError):
        print("Error: no pattern match found or file format wrong, value set to 0")
        value = 0.0

    vals = np.loadtxt(f"./Output/{i}", delimiter=",")[:, 0]

    to_append = pd.DataFrame(vals, columns=["state"])
    to_append["param"] = value

    joined.append(to_append)

runs_array = pd.concat(joined)
trajec_melt = runs_array.melt(id_vars="param", value_vars="state")


# %% convergence checks defining functions


def linreg(x, y): ## snippet stolen from geeksforkeeks
    x = np.array(x)
    y = np.array(y)
    
    n = np.size(x) 
     
    mx, my = np.mean(x), np.mean(y) 
  
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
  
    b1 = ssxy / ssxx 
    b0 = my - b1*mx 
  
    return(b0, b1) 

def estimate_q(trajec, loc=None, regwin = 10):
    
    x = np.arange(len(trajec) - 1)
    y = trajec 
    
    if loc != None:
        # regression window
        x = x[loc-regwin: loc+regwin+1]
        print(f"estimating q for +- {regwin} around loc: {loc}")
        y = trajec[loc-regwin: loc+regwin+1]
    
    y, x = np.asarray([y, x])

     
    idx = np.isfinite(x) & np.isfinite(y)
    
    assert x[idx].ndim == y[idx].ndim, "arrays different dimensions"
    
    #line = np.polyfit(x[idx], y[idx], 1)  # fit degree 1 polynomial
    line = linreg(x, y)
    q= line[1]
    
    #q = np.exp(line[0])  # find q
    rate = -q/(trajec[loc]-1)
    
    return rate, line, y, x 

#test = estimate_q(eps_trunc)


# from original script
def find_root(seq):

    root = 0
    index = 0
    for i in seq:
        if abs(i) < abs(seq[root]):
            root = index
        index = index + 1

    return root

#credit to norok2 from stackoverflow
def find_inflection(seq):
    
    smooth = gaussian_filter1d(seq, 600)
    
    #second derivative
    d2 = np.gradient(np.gradient(smooth))
    
    #find points where sign changes
    infls = np.where(np.diff(np.sign(d2)))[0]
    
    inf_min = 5000
        
    for i in infls:
        
        if i < inf_min:
            continue
        else: 
            inf_ind = i
            break
    
    if len(infls) == 0:
        return False 
    
    print(inf_ind)
    assert inf_ind > inf_min and inf_ind < 20000, "inflection point calculation failed"

    return inf_ind 

def calc_eps_diff(x_n, x_n_1):
    eps_k = abs(x_n-x_n_1)
    return eps_k


def calc_eps_trunc(x_n, x_star):
    eps_k = abs(x_n-x_star)
    return eps_k


# %% calculating epsilon for trajectories rates

trajec_eps = trajec_melt.groupby("param")

#sorry that this is running so slow I will make it faster if I get time#
#runtime will be around 15 minutes for 1 parameter sweep

#list to hold convergence rates
rates_list = []
inflection_list = []

for group_name, trajec in trajec_eps:
    
    root_i = trajec.index[-1]
    
    inflection_x = find_inflection(list(trajec["value"]))
    
    #checking if trajectory isn't a constant 
    if inflection_x:
       # print(inflection_x)
        
        inflection_list.append(inflection_x)
        
        seq_calc_q = trajec_melt[trajec.index[0]:root_i+1]["value"]
        
        rate_to_append = [trajec.iloc[1]["param"], 
                          estimate_q(list(seq_calc_q), loc=inflection_x)[0]]
    else:
        rate_to_append = [trajec.iloc[1]["param"],0] 
        
    rates_list.append(rate_to_append)
    
#df to hold convergence rates
rates = pd.DataFrame(rates_list, columns=["param", "rate"])



# %% plotting functions
#change plot features here
sns.set_theme() 
#set the theme parameters for seaborn
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', "axes.grid": True,
            "grid.color": 'black', 'grid.linestyle': 'dotted', "axes.edgecolor": "black", "patch.edgecolor": "black",
            "patch.linewidth": 0, "axes.spines.bottom": True, 
            "grid.alpha": 0.5, "xtick.bottom": True, "ytick.left": True})

    
#takes data frame of convergence rates for each parameter sweep 
#Rate here
def rate_plot(rates_df, parameter):
    ax = sns.scatterplot(data = rates_df, x ="param", y="rate")
    
    ax.set_ylabel('rate(x1000)')
    ax.set_xlabel('stubbornness') #f"{parameter}"
    #ax.set_title(f"convergence_rate_{scenario}_{parameter}")
    
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(f"../Figs/convergence_rate_{scenario}_{parameter}_v3.png", 
                 bbox_inches='tight', dpi = 300)
    # ax.patch.set_edgecolor('black')  
    # ax.patch.set_linewidth('1') 
    plt.show()

def reg_plot(epsilon_diffs, loc=None):

    q, vals, y, x = estimate_q(epsilon_diffs, loc)

    #xs = list(range(len(y)))
    xs = x

    trendpoly = np.poly1d(vals)

    fig, ax = plt.subplots()
    ax.plot(xs, y)
    ax.plot(xs, trendpoly(xs))
    # ax.set_yscale('log')
    plt.xlabel('k')
    plt.ylabel(r"$\delta_epsilon$")
    plt.title("q = {} convergence".format(q))
    plt.xlim([0,50])
    plt.show()

#Trajec here
def trajec_plot(trajecs):
    #trajecs["param"] = trajecs["value"].astype
    ax = sns.lineplot(data = trajecs, x = "idx", y="value", hue = "param")
    ax.set_ylabel('cooperativity')
    ax.set_xlabel("time [timestep / system size]")
    #ax.set_legend(title=f'{parameter}')
    #ax.set_title(f"trajectories_{scenario}_{parameter}")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    #add label names according to values (see "param") and variable name (see Methods paper)
    ax.legend(loc = 'lower right', fontsize=12, labels = ['\u03A6 = 0.01', '\u03A6 = 0.03', '\u03A6 = 0.06', '\u03A6 = 0.08', '\u03A6 = 0.10'] ) #['\u03A6 = 0.00', '\u03A6 = 0.01', '\u03A6 = 0.02', '\u03A6 = 0.03', '\u03A6 = 0.06', '\u03A6 = 0.08', '\u03A6 = 0.10'];labels = ['\u03A6 = 0.01', '\u03A6 = 0.03', '\u03A6 = 0.06', '\u03A6 = 0.08', '\u03A6 = 0.10']
    ax.set_xlim([0,50])
    # ax.set_ylim([-1,3])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig(f"../Figs/trajec_{scenario}_{parameter}_v3.png", bbox_inches='tight', dpi = 300) 
    plt.show()
    
def epsilon_plot(epsilons):
    sns.lineplot(data = epsilons, x = "idx", y="value", hue = "param")
    plt.xlabel('t')
    plt.ylabel(r"$\epsilon$")
    #plt.title("q = {} convergence".format(estimate_q(epsilons)[0]))
    plt.show()
# %% plotting

#reg_plot(seq_calc_q, loc = inflection_x)

# checking sequence
#plt.plot(trajec["value"])

#filtering for custom rates
#rate_params = ["0.01", "0.1", "0.32", "0.55", "0.78"]


rates["param"] = pd.to_numeric(rates["param"], downcast="float")
rates["rate"] = rates["rate"]*1000
#filter out dataframe based on above parameters, uncomment line below if filtering
#rates = rates[rates.param.astype(str).isin(rate_params)]
#convergence rates
rate_plot(rates, parameter)


#filtering params manually
#vals = ["0.01", "0.10", "0.32", "0.55", "0.78"]
#filters out dataframe for above parameters, uncomment line below if filtering
#trajec_melt = trajec_melt[trajec_melt.param.astype(str).isin(vals)]



#fixing indexes for dataframes so they plot properly
trajec_melt['idx'] = trajec_melt.groupby('param').cumcount() + 1
trajec_melt['idx'] = trajec_melt["idx"]/1089

#avg_cooperation trajectories
trajec_plot(trajec_melt)


#plots the average errors of trajectories for each point x_i for all x in trajec
#epsilon_plot(trajec_melt_eps)


