#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:38:04 2024

@author: leanaking
"""
import abagen
import numpy as np
import pandas as pd
import pickle
import csv
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
import seaborn as sns
import re


#%%
# Using maedbh's package
# not really sure how to do the class thing yet
import transcriptomics
from transcriptomics import data #process_gene_expression
from transcriptomics.constants import Defaults

from transcriptomics import gec_functions_preprocess as preproc 
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures 

#%%
ahba_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid/AHBA/'
sub_dir = '/Users/leanaking/freesurfer/subjects/AHBA/'
project_dir = '/Users/leanaking/Projects/cerebellum_transcriptomics/'

atlas_dir = project_dir + '/external/atlas_templates/'
fig_dir = project_dir + 'leana_folder/figures/'
data_dir = project_dir + 'leana_folder/data/'


#%% Get gene gradient lists
# using original gene set 
# basically replicate dendrogram w/ heatmap 

atlas='SUIT-10'
which_genes='top_bs'
percentile=1 
remove_outliers=True 
atlas_other="MDTB-10" 
normalize=True
# Get df 
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
df = df.set_index(df.index)
# Get clustering 
method='ward' 
metric='euclidean'
orientation='top'
color_leaves=True
set_link_color_palette(['b', 'r', 'y', 'm'])
plt.rcParams['lines.linewidth'] = 2
# Plot Dendrogram
plt.figure(num=1, figsize=[30, 10])
R = dendrogram(
    Z=linkage(df, method, metric),
    orientation=orientation,
    get_leaves=True,
    color_threshold=20.0,
    labels=df.index.to_list(),
    distance_sort='ascending',
    above_threshold_color='black'
)
plt.tick_params(axis='x', which='major', labelsize=12)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.ylabel('Euclidean Distance', fontsize=20)
plt.rcParams['lines.linewidth'] = 6
plt.tight_layout()
plt.savefig(fig_dir+'review_figs/dendo_'+atlas+'_per'+str(percentile)+'.png', bbox_inches="tight", dpi=300)



# Get heatmap - double check it matches with original figure 
dataframe = df.reindex(R['ivl'])
dataframe = dataframe.reindex(index=dataframe.index[::-1])
# plot
plt.figure(num=1, figsize=[7,40])
ax = sns.heatmap(
        dataframe, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(220, 20, n=7), # sns.color_palette("YlOrRd", 10)
        square=False, 
        cbar=True,
        #cbar_kws={"shrink": 0.9},
        yticklabels=True
    )
ax.tick_params(axis='both', which='major', labelsize=16)
ax.yaxis.label.set_size(30)
plt.savefig(fig_dir+'review_figs/heatmap_'+atlas+'_per'+str(percentile)+'.png', bbox_inches="tight", dpi=300)


keys_to_include = ['ivl', 'leaves_color_list']
clust_df = pd.DataFrame({key: R[key] for key in keys_to_include})
clust_df = clust_df.set_index('ivl')
clust_df.index.names = ['Index']

# create df to save 
clust_all_df = clust_df.join(dataframe, how='outer')
clust_all_df = clust_all_df.reset_index()
clust_all_df = clust_all_df.rename(columns={'index': 'gene_name', 'leaves_color_list': 'clust_label'})
clust_all_df = clust_all_df.sort_values('clust_label')
# save 
clust_all_df.to_csv(data_dir+'raster_clust_df_top1_bs_2clust.csv')


