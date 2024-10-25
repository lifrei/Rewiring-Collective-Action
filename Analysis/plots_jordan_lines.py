# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:56:55 2024

@author: everall
"""

import seaborn as sns
import sys
import os
import re
import numpy as  np
import pandas as pd
import seaborn as sns
from statistics import stdev, median, mean
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.ticker import FuncFormatter

date = date.today()
#%% setting file params

file_list = os.listdir("../Output")
file_extension = ".csv"

#%% Default run - Importing model output

# get list of relevant output files
file_list = list(filter(
    lambda x: file_extension in x and "default" in x, os.listdir("../Output")))

file_list = {i:file_list[i] for i in range(len(file_list))}
print(file_list)

file_index = int(input("enter index: "))
#%%
# Set your t_max value here
t_max = 1000

id_vars = ['t', 'scenario', 'rewiring', 'type']
default_run = pd.read_csv(os.path.join("../Output", file_list[file_index]))
#default_run = default_run.drop(default_run.columns[0], axis=1)
print("Unique scenarios:", default_run['scenario'].unique())

default_run['rewiring'] = default_run['rewiring'].fillna('none')
default_run['scenario'] = default_run['scenario'].fillna('none')
print(default_run['scenario'].unique())  # Confirm 'none' is now included


default_r_l = pd.melt(default_run, id_vars=id_vars, var_name='measurement', value_name='value')
default_r_l['scenario_grouped'] = default_r_l['scenario'].str.cat(default_r_l['rewiring'], sep='_')
default_r_l = default_r_l.drop(columns=['scenario', 'rewiring'])
default_r_l['value'] = pd.to_numeric(default_r_l['value'], errors='coerce')

print("Unique measurements:", default_r_l['measurement'].unique())

#data = default_r_l[default_r_l['measurement']=='avg_state']
data = default_r_l[default_r_l['measurement'].isin(['avg_state', 'std_states'])]
print(data.shape)  # Check the size of the filtered dataframe
print(data['measurement'].unique())

# Filter data based on t_max
data = data.drop(data[data['t'] > t_max].index)

#assert 1 == 0
#%% Compare All lineplot
sns.set(style="ticks", font_scale=1.5)


def truncate_labels(x, pos):
    return f'{str(x)[:2]}'  # Format with two decimal places

g = sns.relplot(
    data=data,
    x="t", y="value",
    hue="scenario_grouped",
    style="measurement",  # Add this line to differentiate by measurement
    col="type", 
    kind="line",
    palette="Set2",
    linewidth=1.5,  # Increased slightly for better visibility
    alpha=0.8,
    markers=False, 
    height=5, aspect=1.1,
    facet_kws=dict(sharex=False)
)

# Customize the line styles
g._legend.get_lines()[::2]  # avg_state lines
for line in g._legend.get_lines()[::2]:
    line.set_linestyle('-')  # solid line for avg_state

g._legend.get_lines()[1::2]  # std_states lines
for line in g._legend.get_lines()[1::2]:
    line.set_linestyle(':')  # dotted line for std_states

# Setting titles and labels
g.set_titles("{col_name}", fontweight='bold', color="#333333")
g.set_axis_labels("Time (t)", "Average Opinion State: x")
g.set(xticks=np.arange(0, t_max + 1, t_max // 5))

# Adding grid lines to each subplot
for ax in g.axes.flat:
    ax.grid(True, linestyle='--', linewidth='0.7', color='gray', alpha=0.7)  # Customize the grid style here
    ax.xaxis.set_major_formatter(FuncFormatter(truncate_labels))
# Custom titles for each column
custom_titles = {"FB": "Facebook", "cl": "Clustered Scale-Free"}
for ax, title in zip(g.axes.flat, g.col_names):
    # Set the custom title if defined, otherwise use the default title
    ax.set_title(custom_titles.get(title, title))

# Removing the automatically generated legend
g._legend.remove()

# Extract handles and labels for the legend
handles, labels = g.axes.flat[0].get_legend_handles_labels()

# Creating a new legend closer to the plot
g.fig.legend(handles=handles, labels=labels, title="Rewiring", loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)

# Adjusting bottom to move the legend closer to the plot
plt.subplots_adjust(bottom=0.04)  # Smaller value moves the legend closer

# Save and show the adjusted plot
plt.gcf().set_size_inches(16, 4)
get_N = file_list[file_index].split("_")[5]

plt.savefig(f"../Figs/Trajectories/all_scenarios_compared_N{get_N}_{date}.pdf", dpi=600, bbox_inches='tight')

plt.show()



#%%



#%% Colour/Heatmap














# def extract_values(fname):
#     # Match the structure and extract the values directly
#     match = re.search(r'/(\w+)_linkif_(\w+)_top_(\w+).csv$', fname)
#     return match.groups() if match else (None, None, None)

# joined = []

# for i in file_list:
#     vals = pd.read_csv(f"../Output/{i}", delimiter=",")
#     extract_values(i)
#     joined.append(vals)
    

# extract_values

# runs_array = pd.concat(joined)
# runs_array.to_csv("runs_array.csv")
# data = runs_array
