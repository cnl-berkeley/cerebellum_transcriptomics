#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:09:25 2023

@author: leanaking
"""
import numpy as np
import pandas as pd
import scipy as sp
import pickle
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns; sns.set()


ahba_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid/AHBA/'
sub_dir = '/Users/leanaking/freesurfer/subjects/AHBA/'
project_dir = '/Users/leanaking/Projects/cerebellum_transcriptomics/'

atlas_dir = project_dir + '/external/atlas_templates/'
fig_dir = project_dir + 'leana_folder/figures/'

    
#%% 1. Get list of gene names to print & put into ToppGene enrichment - https://toppgene.cchmc.org/enrichment.jsp
# want GO categories: molecular, biological, and cellular 
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures
from transcriptomics.constants import Defaults 

atlas = "MDTB-10-subRegions"

# 1. Load 1% gene data for MDTB parc
top1_df = pd.read_csv(Defaults.PROCESSED_DIR / 'expression-alldonors-MDTB-10-subRegions-top_bs-1.csv')
gene_list = list(top1_df.columns.values)
del gene_list[-6:]

for gene in gene_list:
    print(gene)


    
#%% 2. Import GO txt files for each category 
        # note that pvalue column is FDR corrected
data_dir = (project_dir + 'leana_folder/data/gene_ontology')    
    

go_df = pd.read_csv(data_dir+'/cerebellum_157G_top1_BS.txt', sep='\t')
go_df['type'] = 'cerebellum'

# Do this for top N? 
# start with just 1% list for now 
df = go_df[['Category', 'Name', 'p-value', 'type']].copy()
df = df.reset_index(drop=True)
# create -log10(pval) column     
df['logpval'] = -np.log10(df['p-value'])
#
thr = 6 # min FDR threshold 
df = df.drop(df[df.logpval < thr].index)

    
    
#%% 3. Pull out sub-categories (biological, molecular, cellular) - turn into function

def get_grouped_pvals(df, df_nodup, grouping): 
    # subset data grouping
    df_subset = df.loc[df['type'] == grouping]
    # initialize new df
    df_grouped = df_nodup[['Category', 'Name']]
    df_grouped['logpval'] = 0.00000
    df_grouped = df_grouped.reset_index(drop=True)
    # fill in pvals 
    for idx, row in df_subset.iterrows():
        name = row['Name']
        val_ind = df_grouped.loc[df_grouped['Name'] == name].index[0]
        df_grouped.at[val_ind, 'logpval'] = row['logpval']
    # remove 'activity' from strings 
    df_grouped.Name = df_grouped.Name.replace('activity', '', regex=True).str.strip()
    df_grouped['Name'] = df_grouped['Name'].str.split(' involved').str[0]
    return df_grouped

def get_subgroups(df):
    molecular_df = df.loc[df['Category'] == 'GO: Molecular Function']
    bio_df = df.loc[df['Category'] == 'GO: Biological Process']
    cellular_df = df.loc[df['Category'] == 'GO: Cellular Component']
    return molecular_df, bio_df, cellular_df
    


# create a new df w/ all GO terms (removing duplicates)
df_nodup = df.drop_duplicates(subset ="Name",
                                keep = 'first', inplace = False)



# for each category at a time, fill in p-values - turn into function 
#df_overlap_all = get_grouped_pvals(df, df_nodup, 'overlap')
df_cerebellum_all = get_grouped_pvals(df, df_nodup, 'cerebellum')




#%% 4. Create histogram plots
## Choose gene category 
# overlap_thr = 8
# cyto_thr = 
condition = 'cerebellum'
molecular_df, bio_df, cellular_df = get_subgroups(df_cerebellum_all)


# get plotting ratios 
mol_len = len(molecular_df)
bio_len = len(bio_df)
cell_len = len(cellular_df)
total_len = mol_len+bio_len+cell_len
cell_ratio = 1.000
multiplier = cell_ratio / (cell_len/total_len)
mol_ratio = multiplier * (mol_len/total_len)
bio_ratio = multiplier * (bio_len/total_len)


## Get plot
names = ['Cellular Component', 'Molecular Function', 'Biological Process'] #1.5, 12
fig, axes = plt.subplots(3,1, figsize=(1.5,12), gridspec_kw={'height_ratios': [cell_ratio, mol_ratio, bio_ratio]}, sharex=True)
# Iterating through axes and names
for name, ax in zip(names, axes.flatten()):
    if name == 'Cellular Component':
        sns.barplot(x="logpval", y="Name", data=cellular_df, orient='h', palette="GnBu_r", ax=ax)
        ax.set_ylabel('Cellular Component', fontsize=14)
        ax.yaxis.label.set_color("darkcyan")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
        ax.set_xlabel('$-log_{10}p(FDR)$', fontsize=14, labelpad=10)    
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', which='both',length=0, labelsize=11) #13
        plt.xlim([5,25]) #10,20 for overlap 
        ax.grid(True, axis='both', color='black', alpha=0.2)
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('black') 
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('black') 
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    if name == 'Molecular Function':
        sns.barplot(x="logpval", y="Name", data=molecular_df, orient='h', palette="BuPu_r", ax=ax)
        ax.set_ylabel('Molecular Function', fontsize=14)
        ax.yaxis.label.set_color("indigo")
        ax.set(xlabel=None)
        plt.yticks(fontsize = 14)
        ax.yaxis.tick_right()    
        ax.tick_params(axis='y', which='both',length=0, labelsize=11)
        ax.grid(True, axis='both', color='black', alpha=0.2)
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    if name == 'Biological Process':
        sns.barplot(x="logpval", y="Name", data=bio_df, orient='h', palette="PuRd_r", ax=ax)
        ax.set_ylabel('Biological Process', fontsize=14)
        ax.yaxis.label.set_color("mediumvioletred")
        #ax.set_xticklabels([])
        ax.set(xlabel=None)
        plt.yticks(fontsize = 14)#14
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', which='both',length=0, labelsize=11)
        ax.grid(True, axis='both', color='black', alpha=0.2)
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
plt.subplots_adjust(hspace=0.025)
plt.savefig(fig_dir + '/gene_ontology_PC1_BS_FDR6_'+condition+'_157G.png', dpi=500, 
            bbox_inches='tight', transparent=True)
plt.show()


