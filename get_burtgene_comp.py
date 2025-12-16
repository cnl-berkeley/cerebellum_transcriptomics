#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:58:59 2023

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

#%% Get gene lists 
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures 

atlas = "MDTB-10"

# start with top 1% to replicate list 
percentile = 1
df = ana.return_thresholded_data(atlas, which_genes='top', percentile=percentile)

# 1. Load 1% gene data for MDTB parc
top1_df = pd.read_csv(project_dir+'processed/expression-alldonors-MDTB-10-top-1.csv')
gene_list = list(top1_df.columns.values)
del gene_list[-6:]



#%% Get Burt cateogries 

old_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid'
#
xls = pd.ExcelFile(old_dir+'/data/burt_gene_categories.xlsx')
df_myelin = pd.read_excel(xls, 'TMC')
df_DS = pd.read_excel(xls, 'DS_C')
df_brain = pd.read_excel(xls, 'Brain', header=None)
df_neuron = pd.read_excel(xls, 'Neuron', header=None)
df_oligo = pd.read_excel(xls, 'Oligodendrocyte', header=None)
df_synapse = pd.read_excel(xls, 'Synaptome', header=None)
df_layers = pd.read_excel(xls, 'Layers')
#
neuron_genes = df_neuron[0].values.tolist()
brain_genes = df_brain[0].values.tolist() 
oligo_genes = df_oligo[0].values.tolist()
synap_genes = df_synapse[0].values.tolist()


# Filter vol processed data to just brain specific genes 
all_genes = neuron_genes + brain_genes + oligo_genes + synap_genes
cat_genes = list(set(brain_genes)) # just gonna use brain genes for now

brain_inds = []
for name in gene_list:
    if name in all_genes: ### filtering to brain-RELATED genes
        ind = gene_list.index(name)
        #print(ind)
        brain_inds.append(ind)
    else:
        pass


bs_gene_name = [gene_list[i] for i in brain_inds]

# only 19 genes out of the 157 are 'brain-specific' ~ 12%
# only 55 genes out of the 157 are in burt categories ~ 35% 





