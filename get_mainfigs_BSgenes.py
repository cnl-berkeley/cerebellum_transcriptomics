#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:14:45 2023

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



#%% Get Burt cateogries 

old_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid'
#
xls = pd.ExcelFile(old_dir+'/data/burt_gene_categories.xlsx')
df_myelin = pd.read_excel(xls, 'TMC')
df_DS = pd.read_excel(xls, 'DS_C') # what is 'DS' ??
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
#layer_genes = df_layers[0].values.tolist()
#myelin_genes = df_myelin[0].values.tolist()

# Filter vol processed data to just brain specific genes 
all_genes = neuron_genes + brain_genes + oligo_genes + synap_genes # + layer_genes + myelin_genes
cat_genes = list(set(brain_genes)) # just gonna use brain genes for now



#%%  Ssve top N% genes - filtered to brain specific  
atlas = 'SUIT-26' #"MDTB-10-subRegions" #'MDTB-10-subRegions' # will need to do for all atlases
which_genes = 'top_bs'
percentile = 1
expression_cleaned = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 
out_name = f"expression-alldonors-{atlas}-{which_genes}-{percentile}.csv"
# get all Genes ranked
ds = preproc._get_differential_stability(atlas)
# subset ds list to burt categories 
ds_allcat = ds[ds.index.isin(all_genes)]
ds_brain = ds[ds.index.isin(brain_genes)]
#
bs_percentile = 8.25
threshold, gene_symbols = preproc._threshold_genes_ds(ds_brain, which_genes='top', percentile=bs_percentile)
# 1% = 157 genes
# 5% = 785 genes
# 10% = 1567 genes
# BS % translations 
# 1% norm = 8.25%
# 5% norm = 41.25%
# 10% norm = 82.4%
# then just do all 'bs' (1902) genes 

expression_thresholded = expression_cleaned[list(gene_symbols) + list(expression_cleaned.filter(regex=("[_].*")).columns)]
expression_thresholded["threshold_type"] = f"{which_genes}_{percentile}%"
# save data 
expression_thresholded.to_csv(Defaults.PROCESSED_DIR / out_name, index=None, header=True)



### Do top 1% for all atlases
atlases = ['Buckner-7', 'Buckner-17', 'Yeo-7'] #"MDTB-10-subRegions", 'MDTB-10"', 
for atlas in atlases:
    which_genes = 'top_bs'
    percentile = 1
    expression_cleaned = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 
    out_name = f"expression-alldonors-{atlas}-{which_genes}-{percentile}.csv"
    # get all Genes ranked
    ds = preproc._get_differential_stability(atlas)
    # subset ds list to burt categories 
    ds_allcat = ds[ds.index.isin(all_genes)]
    ds_brain = ds[ds.index.isin(brain_genes)]
    #
    bs_percentile = 8.25
    threshold, gene_symbols = preproc._threshold_genes_ds(ds_brain, which_genes='top', percentile=bs_percentile)
    expression_thresholded = expression_cleaned[list(gene_symbols) + list(expression_cleaned.filter(regex=("[_].*")).columns)]
    expression_thresholded["threshold_type"] = f"{which_genes}_{percentile}%"
    # save data 
    expression_thresholded.to_csv(Defaults.PROCESSED_DIR / out_name, index=None, header=True)




#%% Replicate Figure 2B plot 
## Get percentile df for input
percentile = 10
atlas='MDTB-10-subRegions'
atlas_other = "MDTB-10" #"MDTB-10-subRegions"
df = ana.return_grouped_data(atlas=atlas, which_genes='top_bs', percentile=percentile, atlas_other=atlas_other, remove_outliers=True, normalize=True)
df = df.set_index(df.index)


## Method 1: using fig2 function
# Can't get leaf color threshold to change??? - otherwise, code works for get different top %s
#figures.fig_2(which_genes='top', percentile=percentile, remove_outliers=True, reorder_labels=True, atlas_other=atlas, normalize=True)



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
        color_threshold=120.0,  # 1% = 35, 5% = 75, # 10% = 120
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
plt.rcParams['lines.linewidth'] = 3
plt.tight_layout()
#
plt.savefig(fig_dir+'Fig2_BS/B_PLOT_dendo_'+atlas+'_per'+str(percentile)+'_BS.png', bbox_inches="tight", dpi=600)





#%% Replicate rest of Figure 2 plots 

## Just learned that figures C-E do not change w/ percentile in method1 (whole plot) above
# will need to plot each sub-fig seperately as well
percentiles = [1, 5, 10]
which_genes = 'top_bs'
atlas_other = "MDTB-10" 
#fig, (ax1, ax2, ax3) = plt.subplots(3)
for percentile in percentiles: 
    df = ana.return_grouped_data(atlas='MDTB-10-subRegions', which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, reorder_labels=True, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas='MDTB-10-subRegions', distance_correct=True)#, ax=ax1)
    plt.savefig(fig_dir+'Fig2_BS/C_heatmap_per'+str(percentile)+'.png', bbox_inches="tight", dpi=600)
    plt.clf()

    # 2D
    atlas_other="MDTB-10"
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-7", atlas_cortex="Yeo-7", which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-7", distance_correct=True, simple_labels=True)# ax = ax2)
    plt.savefig(fig_dir+'Fig2_BS/D_heatmap_per'+str(percentile)+'.png', bbox_inches="tight", dpi=600)
    plt.clf()
    
    # 2E
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-17", atlas_cortex="Yeo-17", which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-17", distance_correct=True, simple_labels=True)#, ax = ax3)
    plt.savefig(fig_dir+'Fig2_BS/E_heatmap_per'+str(percentile)+'.png', bbox_inches="tight", dpi=600)
    plt.clf()

    

#%% Replicate Figure 3 plots

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
plt.savefig(fig_dir+'Fig3_BS/B_dendo_'+atlas+'_per'+str(percentile)+'_BS.png', bbox_inches="tight", dpi=600)



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
plt.savefig(fig_dir+'Fig3_BS/B_raster_'+atlas+'_per'+str(percentile)+'_BS.png', bbox_inches="tight", dpi=600)



# 3c
ax = visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig(fig_dir+'Fig3_BS/C_corrmat_'+atlas+'_per'+str(percentile)+'_BS.png', bbox_inches="tight", dpi=600)




# 3d
plt.figure(num=1, figsize=[8,6])
visualize.dendrogram_plot(df.T, color_leaves=False)
plt.tick_params(axis='x', which='major', labelsize=15, rotation=45)
plt.tick_params(axis='y', which='major', labelsize=25)
plt.tight_layout()
#plt.yaxis.label.set_size(20)
plt.savefig(fig_dir+'Fig3_BS/D_dendo_'+atlas+'_per'+str(percentile)+'_BS.png', bbox_inches="tight", dpi=600)




# Get variance explained by first PC
percentile = 1
atlas='SUIT-10'
which_genes='top_bs'
remove_outliers=True
atlas_other="MDTB-10-subRegions"  # hmmm the subRegions option still gives the same PC1 and PC2 pattern, but 3D plots are very different 
normalize=True
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, 
                             atlas_other=atlas_other, remove_outliers=remove_outliers, 
                             normalize=normalize)

ana._variance_explained(df, pcs=[1]) #not zero-indexed

# 3e
group_pcs=False
cmap, labels = visualize._make_colormap(atlas)
#mpl.cm.register_cmap("mycolormap", cmap)
cpal = sns.color_palette("mycolormap", n_colors=len(labels))
# gets the correct color for each region (only a problem when there are missing regions)
labels, cpal_reordered = visualize._reorder_colors_x_axis(df, cpal)
#labels = df.columns
u, s, vt, pcs = ana._compute_svd(df)
# zero index the pcs
# plot
plt.figure(num=2, figsize=[10,6])
ax = sns.barplot(x=labels, y=vt[1, :], palette=cpal, alpha=0.7) # indexs into list
# ax.set_xticks(labels) # rotation=90
ax.set_ylabel('PC2 Loading')
ax.set_xlabel('')
ax.tick_params(axis='x', which='major', labelsize=20, rotation=45)
ax.tick_params(axis='y', which='major', labelsize=25)
ax.yaxis.label.set_size(20)
plt.tight_layout()
plt.savefig(fig_dir+'Fig3_BS/E_pc3Bar_'+atlas+'_per'+str(percentile)+'_BS.png', bbox_inches="tight", dpi=600)





#%%Get images for other plots
def plotting_style():
    # fig = plt.figure(num=2, figsize=[20,8])
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 6
    plt.rc('xtick', labelsize=14)     
    plt.rc('ytick', labelsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'svg'
    plt.rc("axes.spines", top=False, right=False) # removes certain axes


plt.clf()

plotting_style()

# 2a
atlas = 'MDTB-10-subRegions'
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot()
visualize.png_plot(filename='MDTB-10-subRegions', ax=ax1)
ax1.axis('off')
plt.savefig(fig_dir+'Fig2_BS/A_'+atlas+'.png', 
            bbox_inches="tight", dpi=600, transparent=True)


#3a
atlas = 'SUIT-10_v2'
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot()
visualize.png_plot(filename='SUIT-10_v2', ax=ax1)
ax1.axis('off')
ax1.yaxis.label.set_size(30)
plt.savefig(fig_dir+'Fig3_BS/A_'+atlas+'.png', 
            bbox_inches="tight", dpi=600, transparent=True)


# 4a
image1="rat_zones"
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot()
visualize.png_plot(filename=image1, ax=ax1)
ax1.axis('off')
plt.savefig(fig_dir+'Fig4_BS/A_'+image1+'.png', 
            bbox_inches="tight", dpi=600, transparent=True)
