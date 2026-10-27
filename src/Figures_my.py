import sys
import matplotlib.pyplot as plt
from src.eval_qa import eval_file
import argparse
import pandas as pd
import torch
import json

def Loss_graph_1(path_input_0, path_input_1 , title_1="Model Loss over Optimization Steps (Normal)", title_2="Model Loss over Optimization Steps (Sharing)", compare_scale=False):
    """
    Plots two graphs side by side for model loss over optimization steps.
    
    Parameters:
    - data_0: DataFrame containing 'global_step', 'train_loss', and 'eval_loss' columns for the first dataset
    - data_1: DataFrame containing 'global_step', 'train_loss', and 'eval_loss' columns for the second dataset
    - title_1: Title for the first graph
    - title_2: Title for the second graph
    """
    path_csv_0 = path_input_0 + "training_progress_scores.csv" 
    path_csv_1 = path_input_1 + "training_progress_scores.csv" 
    data_0 = pd.read_csv(path_csv_0)
    data_1 = pd.read_csv(path_csv_1)
    
    # Filter the rows for eval_loss and train_loss separately for both datasets
    data_0_eval_loss_df = data_0[data_0["eval_loss"] != -1]
    data_0_train_loss_df = data_0[data_0["train_loss"] != -1]
    data_1_eval_loss_df = data_1[data_1["eval_loss"] != -1]
    data_1_train_loss_df = data_1[data_1["train_loss"] != -1]

    # Create the figure and axis objects
    fig, axs = plt.subplots(1, 2, figsize=(13, 5),  sharey=True)
    fig.suptitle('Loss Composition', fontsize=13, fontweight='bold')
    if compare_scale:
        axs[1].set_xlabel('Optimization Step ')
        #axs[1].set_xscale('log')
        axs[0].set_xscale('log')  # Set X-axis to log scale
        axs[0].set_xlabel('Optimization Step (Log Scale)')
    else:
        axs[0].set_xscale('log')  # Set X-axis to log scale
        axs[0].set_xlabel('Optimization Step (Log Scale)')
        axs[1].set_xlabel('Optimization Step (Log Scale)')
        axs[1].set_xscale('log')  # Set X-axis to log scale
                          

    # First graph: Model loss over optimization steps (Normal)
    axs[0].plot(data_0_train_loss_df["global_step"], data_0_train_loss_df["train_loss"], 'o-', label='Train Loss', color='r')
    axs[0].plot(data_0_eval_loss_df["global_step"], data_0_eval_loss_df["eval_loss"], 's-', label='Eval Loss', color='g')
    #axs[0].set_xscale('log')  # Set X-axis to log scale
    #axs[0].set_xlabel('Optimization Step (Log Scale)')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title(title_1)

    # Second graph: Model loss over optimization steps (Sharing)
    axs[1].plot(data_1_train_loss_df["global_step"], data_1_train_loss_df["train_loss"], 'o-', label='Train Loss', color='r')
    axs[1].plot(data_1_eval_loss_df["global_step"], data_1_eval_loss_df["eval_loss"], 's-', label='Eval Loss', color='g')
    
    #axs[1].set_xscale('log')  # Set X-axis to log scale
    #axs[1].set_xlabel('Optimization Step (Log Scale)')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title(title_2)
    #axs[1].set_xlim(2000, 10**5)  # Set the x-axis range to start from 10^3
    #axs[0].set_xlim(2000, 10**6)

    plt.tight_layout()  # Adjust layout to make room for the main title
    plt.show()




# Data loaded from the previous step

def Acc_graph(data , data_sharing , title_1= 'Model Accuracy over Optimization Steps (Normal)',title_2= 'Model Accuracy over Optimization Steps (Sharing)', compare_scale=False ):
    
    optimization_steps = [int(checkpoint.split('-')[1]) for checkpoint, _ in data]
    train_id = [dict(results)['train_inferred'] for _, results in data]   #inferred train-> iid as no train on ood obviously!
    test_id = [dict(results)['test_inferred_iid'] for _, results in data]
    test_ood = [dict(results)['test_inferred_ood'] for _, results in data]

    # Sample Data (replace these with your actual data)

    # Create the figure and axis objects
   
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    # title figure
    fig.suptitle('Accuracy Composition', fontsize=13, fontweight='bold' )


    if compare_scale:
        axs[1].set_xlabel('Optimization Step ')
        #axs[1].set_xscale('log')
        axs[0].set_xscale('log')  # Set X-axis to log scale
        axs[0].set_xlabel('Optimization Step (Log Scale)')
    else:
        axs[0].set_xscale('log')  # Set X-axis to log scale
        axs[0].set_xlabel('Optimization Step (Log Scale)')
        axs[1].set_xlabel('Optimization Step (Log Scale)')
        axs[1].set_xscale('log')  # Set X-axis to log scale

    # First graph
    axs[0].plot(optimization_steps, train_id, 'o-', label='Train (ID)', color='r')
    axs[0].plot(optimization_steps, test_id, 's-', label='Test (ID)', color='g')
    axs[0].plot(optimization_steps, test_ood, '^-', label='Test (OOD)', color='b')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True)  # Enable grid lines for the first graph
    # title
    axs[0].set_title(title_1, fontsize=13)

    # Second graph (adjusting accuracy curves for the second graph)
    optimization_steps = [int(checkpoint.split('-')[1]) for checkpoint, _ in data_sharing]
    train_id = [dict(results)['train_inferred'] for _, results in data_sharing]
    test_id = [dict(results)['test_inferred_iid'] for _, results in data_sharing]
    test_ood = [dict(results)['test_inferred_ood'] for _, results in data_sharing]

    # title
    axs[1].set_title(title_2, fontsize=13)

    axs[1].plot(optimization_steps, train_id, 'o-', label='Train (ID)', color='r')
    axs[1].plot(optimization_steps, test_id, 's-', label='Test (ID)', color='g')
    axs[1].plot(optimization_steps, test_ood, '^-', label='Test (OOD)', color='b')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)  # Enable grid lines for the first graph

    plt.tight_layout()
    plt.show()


################


import matplotlib.pyplot as plt
#import seaborn as sns

def Acc_1(data, log=True):
    optimization_steps = [int(checkpoint.split('-')[1]) for checkpoint, _ in data]
    train_id = [dict(results)['train_inferred'] for _, results in data]  
    test_id = [dict(results)['test_inferred_iid'] for _, results in data]
    test_ood = [dict(results)['test_inferred_ood'] for _, results in data]

    # Use Accent colormap
    accent_colors = plt.get_cmap("Accent").colors

    set3_cmap = plt.get_cmap("Set3")
    set1_cmap = plt.get_cmap("Set2")
    train_color = set3_cmap(8) #'#A9A9A9' #'#A9A9A9' #accent_colors[6]  # Soft Gray
    test_id_color =  accent_colors[1] #set3_cmap(2) #accent_colors[3]  # Violet
    test_ood_color = set1_cmap(5)# set3_cmap(11) #set1_cmap(5) #"#E41A1C" #'#FFD700' #accent_colors[1]  # Yellow

    # Create the figure
    plt.figure(figsize=(8, 5))

    # Plot the data with Accent colormap colors
    plt.plot(optimization_steps, train_id, 'o-', label='Train (ID)', color=train_color)
    plt.plot(optimization_steps, test_id, 's-', label='Test (ID)', color=test_id_color)
    plt.plot(optimization_steps, test_ood, '^-', label='Test (OOD)', color=test_ood_color)

    # Labels and title
    plt.ylabel('Accuracy')
    plt.xlabel('Optimization Step (Log Scale)' if log else 'Optimization Step')
    #plt.title('Accuracy Composition', fontsize=13, fontweight='bold')

    # Set log scale if required
    if log:
        plt.xscale('log')

    # Legend and grid
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

######################################

def dic_scores( dir, check = False, fn = 'all_items.json',file_name = 'Accuracy_Loss.ipynb'):

    #sys.argv = ['Accuracy_Loss.ipynb', '--dir', path_es_1_SH, '--fn', 'all_items.json'] # (sys.argv this line is to use it in notebook only
    sys.argv = [file_name , '--dir', dir, '--fn', fn]

    #this is used to run the script from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, type=str, required=True, help="Input file dir.")
    parser.add_argument("--fn", default='all_items.json', type=str, help="")
    parser.add_argument("--partition_atomic", action="store_true", help="")
    args = parser.parse_args()
    args.dir, args.fn, args.partition_atomic

    scores_dict = eval_file(args.dir, args.fn, args.partition_atomic) # list of tuple (folder_name, res) = (checkpoint, "sringa di risultati"))

    if check:
        temp = []
        # temp = lista di tuple (folder_name, res) = (checkpoint, "sringa di risultati"))
        for (folder_name, val) in scores_dict:
            temp.append((folder_name, "; ".join(["{}: {}".format(t, res) for (t, res) in val])))

        for (folder_name, res) in temp:
            print(folder_name, "|", res)
            #continue

    return scores_dict