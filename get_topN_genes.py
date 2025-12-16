#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:38:42 2023

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
from sklearn import cluster
from scipy.spatial.distance import cdist
from nilearn import image
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns


#%%
# Using maedbh's package
# not really sure how to do the class thing yet
import transcriptomics
from transcriptomics import data #process_gene_expression
from transcriptomics.constants import Defaults


#%%
ahba_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid/AHBA/'
sub_dir = '/Users/leanaking/freesurfer/subjects/AHBA/'
project_dir = '/Users/leanaking/Projects/cerebellum_transcriptomics/'

atlas_dir = project_dir + '/external/atlas_templates/'
fig_dir = project_dir + 'leana_folder/figures/'


#%%
# Load parcellated abagen data for MDTB parc
colinfostr = ['region_num', 'region_id', 'colours', 'donor_id', 'donor_num', 'sample_counts']
# From here, was data averaged across donors or was an abagen function used? 
# - use 'return_grouped_data' function
test_df = pd.read_csv(project_dir+'interim/expression-alldonors-MDTB-10-cleaned.csv')



#%% Repeat maeve's method to get top n% of genes - just practice loading data for now
from transcriptomics import gec_functions_preprocess as preproc 
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures 

atlas = "MDTB-10-subRegions"

# start with top 1% to replicate list 
percentile = 50 #= 157 genes
test_df = ana.return_thresholded_data(atlas, which_genes='top', percentile=percentile) #num col -5 

# 1. Load 1% gene data for MDTB parc
#top1_df = pd.read_csv(project_dir+'processed/expression-alldonors-MDTB-10-top-1.csv')

# 1% = 157 genes
# 5% = 785 genes
# 10% = 1567 genes
# 25% = 3916 genes
# 50% = 7831 genes

#%% Replicate Figure 2 plots - i.e. repeating sensitivity analysis

## Get percentile df for input
percentile = 50
atlas='MDTB-10-subRegions'
atlas_other = "MDTB-10" #"MDTB-10-subRegions"
df = ana.return_grouped_data(atlas=atlas, which_genes='top', percentile=percentile, atlas_other=atlas_other, remove_outliers=True, normalize=True)
df = df.set_index(df.index)


## Method 1: using fig2 function
# Can't get leaf color threshold to change??? - otherwise, code works for get different top %s
# b. dendrograms are not replicated with this function, but replicated w/ method 4 below
fig = figures.fig_2(which_genes='top', percentile=percentile, remove_outliers=True, reorder_labels=True, atlas_other=atlas, normalize=True)

plt.savefig((fig_dir+'Fig2/totalfig_'+atlas+'_per'+str(percentile)+'.png'), bbox_inches="tight", dpi=300)


## Method 4 --- THIS ONE WORKS! ^ pair this we/ corr plots from method above
# get dataframe input
dataframe = df.T
dataframe = visualize._reorder_dendrogram_leaves(dataframe)

# Plot dendrogram seperately 
set_link_color_palette(['b', 'r', 'y', 'm'])
plt.figure(num=1, figsize=[30,10])
R = dendrogram(
        Z=linkage(dataframe, method='ward', metric='euclidean'),
        orientation='top',
        color_threshold=150.0,  # 1%=35, 5%=45, 10%=60 , 25%=100 , 50%=150
        get_leaves=True,
        labels=dataframe.index.to_list(),
        distance_sort='ascending',
        above_threshold_color='black')
plt.xlabel('')
#plt.ylabel(f"{metric.capitalize()} Distance")
plt.tick_params(axis='x', which='major', labelsize=40, rotation=45)
plt.tick_params(axis='y', which='major', labelsize=40)
#ax2.axis('off')
#plt.text(x_pos+0.06, y_pos, 'B', transform=ax2.transAxes, fontsize=60,
#    verticalalignment='top')
plt.ylabel('Euclidean Distance', fontsize=50)
plt.tight_layout()
#
plt.savefig(fig_dir+'Fig2/B_dendo_'+atlas+'_per'+str(percentile)+'.png')



## Just learned that figures C-E do not change w/ percentile in method1 (whole plot) above
# will need to plot each sub-fig seperately as well
percentiles = [1, 5, 10, 25, 50]
#fig, (ax1, ax2, ax3) = plt.subplots(3)
for percentile in percentiles: 
    df = ana.return_grouped_data(atlas='MDTB-10-subRegions', which_genes='top', atlas_other=atlas_other, percentile=percentile, reorder_labels=True, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas='MDTB-10-subRegions', distance_correct=True)#, ax=ax1)
    plt.savefig(fig_dir+'Fig2/C_heatmap_'+atlas+'_per'+str(percentile)+'.png')
    plt.clf()

    # 2D
    atlas_other="MDTB-10"
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-7", atlas_cortex="Yeo-7", which_genes='top', atlas_other=atlas_other, percentile=percentile, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-7", distance_correct=True, simple_labels=True)# ax = ax2)
    plt.savefig(fig_dir+'Fig2/D_heatmap_'+atlas+'_per'+str(percentile)+'.png')
    plt.clf()
    
    # 2E
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-17", atlas_cortex="Yeo-17", which_genes='top', atlas_other=atlas_other, percentile=percentile, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-17", distance_correct=True, simple_labels=True)#, ax = ax3)
    plt.savefig(fig_dir+'Fig2/E_heatmap_'+atlas+'_per'+str(percentile)+'.png')
    plt.clf()

    

#%% Replicate FIgure 3 plots


## Get percentile df for input
percentile = 1
atlas='SUIT-10'
which_genes='top'
remove_outliers=True
atlas_other="MDTB-10"
normalize=True



# 3bi
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
df = df.set_index(df.index)
method='ward' 
metric='euclidean'
orientation='left'
color_leaves=False
# plot
plt.figure(num=1, figsize=[10,40])
R = dendrogram(
    Z=linkage(df, method, metric),
    orientation=orientation,
    get_leaves=True,
    labels=df.index.to_list(),
    distance_sort='ascending',
    color_threshold=0,
    above_threshold_color='black'
    )
#plt.ylabel(f"{metric.capitalize()} Distance")
plt.tick_params(axis='x', which='major', labelsize=40)
plt.tick_params(axis='y', which='major', labelsize=20)
#ax2.axis('off')
#plt.text(x_pos+0.06, y_pos, 'B', transform=ax2.transAxes, fontsize=60,
#    verticalalignment='top')
plt.xlabel('Euclidean Distance', fontsize=50)
plt.rcParams['lines.linewidth'] = 6
plt.tight_layout()
#
plt.savefig(fig_dir+'Fig3/B_dendo_'+atlas+'_per'+str(percentile)+'.png')



# 3bii
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
dataframe = df.set_index(df.index)
R = dendrogram(
    Z=linkage(dataframe, method, metric),
    labels=df.index.to_list(),
    distance_sort='ascending',
    )
dataframe = dataframe.reindex(R['ivl'])
dataframe = dataframe.reindex(index=dataframe.index[::-1])
plt.clf()

# plot
plt.figure(num=1, figsize=[7,40])
ax = sns.heatmap(
        dataframe, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(220, 20, n=7), # sns.color_palette("YlOrRd", 10)
        square=False, 
        cbar=True,
        #cbar_kws={"shrink": 0.9},
        yticklabels=False
    )
ax.tick_params(axis='both', which='major', labelsize=30)
ax.yaxis.label.set_size(30)
plt.savefig(fig_dir+'Fig3/B_raster_'+atlas+'_per'+str(percentile)+'.png')


###

# 3c
ax = visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig(fig_dir+'Fig3/C_corrmat_'+atlas+'_per'+str(percentile)+'.png')



# 3d
plt.figure(num=1, figsize=[8,6])
visualize.dendrogram_plot(df.T, color_leaves=False)
plt.tick_params(axis='x', which='major', labelsize=15, rotation=45)
plt.tick_params(axis='y', which='major', labelsize=25)
#plt.yaxis.label.set_size(20)
plt.tight_layout()
plt.savefig(fig_dir+'Fig3/D_dendo_'+atlas+'_per'+str(percentile)+'.png')


group_pcs=False
pcs=[1]
cmap, labels = visualize._make_colormap(atlas)
cpal = sns.color_palette("mycolormap", n_colors=len(labels))
# gets the correct color for each region (only a problem when there are missing regions)
labels, cpal_reordered = visualize._reorder_colors_x_axis(df, cpal)
#labels = df.columns
u, s, vt, _ = ana._compute_svd(df)
# zero index the pcs
pcs = [x-1 for x in pcs]
# plot
plt.figure(num=2, figsize=[10,6])
ax = sns.barplot(x=labels, y=vt[pcs[0], :], palette=cpal, alpha=0.7) # indexs into list
# ax.set_xticks(labels) # rotation=90
ax.set_ylabel(f'PC{pcs[0]+1} Loading')
ax.set_xlabel('')
ax.tick_params(axis='x', which='major', labelsize=20, rotation=45)
ax.tick_params(axis='y', which='major', labelsize=25)
ax.yaxis.label.set_size(20)
plt.tight_layout()
plt.savefig(fig_dir+'Fig3/E_pcBar_'+atlas+'_per'+str(percentile)+'.png')



#%% Replicate Figure 4 plots

image1="rat_zones"
image2="MDTB-10-subRegions-transcriptomic"
atlas='MDTB-10-subRegions' 
which_genes='top_bs' 
percentile=1
remove_outliers=True 
atlas_other="MDTB-10"
normalize=True


# 4b
visualize.png_plot(filename=image2)
plt.savefig(fig_dir+'Fig4/B_cortexplot_'+image2+'_per'+str(percentile)+'.png')
