import sys
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FuncFormatter
from datetime import date

# Set the seaborn theme parameters


# Global settings
FONT_SIZE = 14
FRIENDLY_COLORS = {
    'static': '#EE7733',      # Orange
    'random': '#0077BB',      # Blue
    'local (similar)': '#33BBEE',    # Cyan
    'local (opposite)': '#009988',   # Teal
    'bridge (similar)': '#CC3311',   # Red
    'bridge (opposite)': '#EE3377',  # Magenta
    'wtf': '#BBBBBB',         # Grey
    'node2vec': '#44BB99'     # Blue-green
}

# Set the seaborn theme parameters
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
sns.set_theme(font_scale=FONT_SIZE/14)  # Default font size is 12, so scale accordingly
sns.set(style="ticks")
sns.set(rc={'axes.facecolor':'white', 
            'font.family': 'Arial',
            'font.size': 14,
            'figure.facecolor':'white', 
            "axes.grid": True,
            "grid.color": 'black', 
            'grid.linestyle': 'dotted', 
            "axes.edgecolor": "black", 
            "patch.edgecolor": "black",
            "patch.linewidth": 0, 
            "axes.spines.bottom": True, 
            "grid.alpha": 0.5, 
            "xtick.bottom": True, 
            "ytick.left": True})


# Define friendly names mapping
friendly_names = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)',
    'bridge_same': 'bridge (similar)',
    'bridge_diff': 'bridge (opposite)',
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}

def find_inflection(seq):
    """Calculate inflection point in trajectory"""
    smooth = gaussian_filter1d(seq, 600)
    d2 = np.gradient(np.gradient(smooth))
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
    
    assert inf_ind > inf_min and inf_ind < 20000, "inflection point calculation failed"
    return inf_ind 

def estimate_convergence_rate(trajec, loc=None, regwin=10):
    """Estimate convergence rate around specified location"""
    x = np.arange(len(trajec) - 1)
    y = trajec 
    
    if loc is not None:
        x = x[loc-regwin: loc+regwin+1]
        y = trajec[loc-regwin: loc+regwin+1]
    
    y, x = np.asarray([y, x])
    idx = np.isfinite(x) & np.isfinite(y)
    
    # Linear regression
    n = np.size(x) 
    mx, my = np.mean(x), np.mean(y) 
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
    b1 = ssxy / ssxx 
    b0 = my - b1*mx 
    
    rate = -b1/(trajec[loc]-1)
    return rate

def calculate_convergence_rates(data):
    """Calculate convergence rates for all scenarios"""
    rates_list = []
    
    # Group by scenario and type
    grouped = data.groupby(['scenario_grouped', 'type'])
    
    for name, group in grouped:
        scenario, topology = name
        
        # Get trajectory for this scenario
        trajectory = group[group['measurement'] == 'avg_state']['value'].values
        
        # Find inflection point
        inflection_x = find_inflection(trajectory)
        
        if inflection_x:
            rate = estimate_convergence_rate(trajectory, loc=inflection_x)
        else:
            rate = 0
            
        # Map to friendly name
        friendly_scenario = friendly_names.get(scenario, scenario)
            
        rates_list.append({
            'scenario': friendly_scenario,
            'topology': topology,
            'rate': rate * 1000  # Scale rate as in original code
        })
    
    return pd.DataFrame(rates_list)




def plot_convergence_rates(rates_df, output_path):
    """Create convergence rate comparison plot with scenarios ordered by median convergence rate,
    using traditional statistical physics markers in black and white"""
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(12, 8))
    
    # Calculate median rate for each scenario to determine order
    scenario_medians = rates_df.groupby('scenario')['rate'].median().sort_values(ascending=False)
    scenario_order = scenario_medians.index.tolist()
    
    # Create a new categorical column with ordered scenarios
    rates_df = rates_df.copy()  # Create a copy to avoid SettingWithCopyWarning
    rates_df['scenario'] = pd.Categorical(rates_df['scenario'], 
                                        categories=scenario_order, 
                                        ordered=True)
    
    # Define traditional statistical physics markers and their labels
    topology_markers = {
        'DPAH': 'x',      # cross
        'cl': '+',        # plus
        'Twitter': '*',   # asterisk
        'FB': '.'         # point
    }
    
    # Create x-coordinates for scenarios
    x_coords = np.arange(len(scenario_order))
    
    # Create main scatter plot with topology markers
    for topology, marker in topology_markers.items():
        mask = rates_df['topology'] == topology
        scenario_data = rates_df[mask]
        
        # Map scenarios to x-coordinates
        scenario_x = [x_coords[list(scenario_order).index(s)] for s in scenario_data['scenario']]
        
        plt.scatter(
            scenario_x,
            scenario_data['rate'],
            marker=marker,
            s=100,  # Marker size
            c='black',  # All markers in black
            label=topology,
            alpha=0.7
        )
    
    # Add median values as horizontal lines
    for i, scenario in enumerate(scenario_order):
        median_rate = scenario_medians[scenario]
        plt.hlines(y=median_rate, xmin=i-0.2, xmax=i+0.2, 
                  colors='red', linestyles='solid', alpha=0.5)
    
    # Customize the plot
    plt.title('Convergence Rates Comparison', pad=20, fontsize=FONT_SIZE)
    plt.xlabel('Rewiring Strategy', labelpad=10, fontsize=FONT_SIZE)
    plt.ylabel('Convergence Rate (×10³)', labelpad=10, fontsize=FONT_SIZE)
    
    # Set x-axis ticks and labels
    plt.xticks(x_coords, scenario_order, 
               rotation=45, 
               horizontalalignment='right',
               rotation_mode='anchor',
               fontsize=FONT_SIZE)
    
    # Create legend for topologies (markers)
    legend_elements = [plt.Line2D([0], [0], marker=marker, color='black', 
                                label=topology, markersize=10, linestyle='None')
                      for topology, marker in topology_markers.items()]
    
    # Add median line to legend
    legend_elements.append(plt.Line2D([0], [0], color='red', alpha=0.5,
                                    label='Median', linestyle='solid'))
    
    # Place legend
    plt.legend(handles=legend_elements, title="Network Topology", 
              loc='upper left', bbox_to_anchor=(0.85, 0.95))
    
    plt.yticks(fontsize=FONT_SIZE)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight', 
                format='pdf',
                metadata={'Creator': "Jordan", 'Producer': None})
    plt.show()
    plt.close()
# # Update friendly names to be more concise for publication
# friendly_names = {
#     'none_none': 'static',
#     'random_none': 'random',
#     'biased_same': 'similar',
#     'biased_diff': 'opposite',
#     'bridge_same': 'bridge-s',
#     'bridge_diff': 'bridge-o',
#     'wtf_none': 'wtf',
#     'node2vec_none': 'n2vec'
# }

def main():
    # Get list of relevant output files
    file_list = [f for f in os.listdir("../Output") if f.endswith(".csv") and "default_run_avg" in f]

    if not file_list:
        print("No suitable files found in the Output directory.")
        exit()

    # Print file list with indices
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")

    # Get user input for file selection
    file_index = int(input("Enter the index of the file you want to plot: "))
    
    # Load the selected file
    data_path = os.path.join("../Output", file_list[file_index])
    raw_data = pd.read_csv(data_path)
    
    # Prepare data for analysis
    id_vars = ['t', 'scenario', 'rewiring', 'type']
    raw_data['rewiring'] = raw_data['rewiring'].fillna('none')
    raw_data['scenario'] = raw_data['scenario'].fillna('none')
    
    # Melt the dataframe
    melted_data = pd.melt(raw_data, id_vars=id_vars, 
                         var_name='measurement', value_name='value')
    melted_data['scenario_grouped'] = melted_data['scenario'].str.cat(
        melted_data['rewiring'], sep='_')
    
    # Calculate convergence rates
    rates_df = calculate_convergence_rates(melted_data)
    
    # Create output directory if it doesn't exist
    os.makedirs("../Figs/Convergence", exist_ok=True)
    
    # Generate output filename
    N = file_list[file_index].split("_")[5]  # Extract N from filename
    output_path = f"../Figs/Convergence/convergence_rates_N{N}_{date.today()}.pdf"
    
    # Create and save the plot
    plot_convergence_rates(rates_df, output_path)
    print(f"Convergence rate plot saved to: {output_path}")
    
    # Print summary statistics with friendly names
    print("\nConvergence Rate Summary Statistics:")
    print(rates_df.groupby(['scenario'])['rate'].describe())
    
    return rates_df

if __name__ == "__main__":
   df_rates = main()