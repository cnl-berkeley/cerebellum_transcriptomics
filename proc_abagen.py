#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:41:24 2023

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


#%%
# Using maedbh's package
# not really sure how to do the class thing yet

import transcriptomics
from transcriptomics import data #process_gene_expression
from transcriptomics.constants import Defaults

#df = data.DataSet.process_gene_expression(atlas='MDTB-10')


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
df = pd.read_csv(project_dir+'interim/expression-alldonors-MDTB-10-cleaned.csv')



#%% Apply clustering to top 1% genes to get the 3 gradient labels 
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures 


# 1. Load 1% gene data for MDTB parc
atlas_other_df = pd.read_csv(project_dir+'processed/expression-alldonors-MDTB-10-top-1.csv')


# 2. replicate the following function to get the input df for fig3.b
# df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
# parameters:
atlas = 'SUIT-10'
which_genes ='top'
percentile = 1
atlas_other = "MDTB-10"
remove_outliers=True
normalize=True

# df proc steps 
# a. using SUIT atlast for this figure
dataframe = pd.read_csv(project_dir+'interim/expression-alldonors-SUIT-10-cleaned.csv')
# b. 
dataframe = ana._remove_outliers(dataframe, atlas)
# c. filter based on genes list from MDTB atlast 
genes = atlas_other_df.filter(regex=("[A-Z0-9].*")).columns
dataframe = dataframe[list(genes) + list(dataframe.filter(regex=("[_].*")).columns)]
# d. group across regions 
dataframe = ana._group_by_region(dataframe)
# e. normalized dataframe
dataframe = ana._center_scale(dataframe.T)

    
# 3. Replicate Fig3.b plot to gene gradient labels     
# load plotting parameters
figures.plotting_style()
fig = plt.figure(figsize=(20, 30))
gs = GridSpec(3, 3, figure=fig)
x_pos = -0.2
y_pos = 1.02

# Plot 
# dendrogram
ax1 = fig.add_subplot(gs[:, 2])
df = dataframe.set_index(dataframe.index)
plt.rcParams['lines.linewidth'] = 2
visualize.dendrogram_plot(df, orientation='left', color_leaves=True, ax=ax1) #color_leaves=False,
ax1.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False)
ax1.axis('off')
ax1.text(x_pos, y_pos, 'B', transform=ax1.transAxes, fontsize=40,
verticalalignment='top')
ax1.yaxis.label.set_size(30)
plt.rcParams['lines.linewidth'] = 6

# raster plot
# ax2 = fig.add_subplot(gs[:, 3]) # original line
f, ax2 = plt.subplots(1, 1, figsize=(5,20)) # modified 
df = dataframe
visualize.raster_plot(df,  ax=ax2)
ax2.tick_params(axis='both', which='major', labelsize=15)
#ax2.text(x_pos, y_pos, 'C', transform=ax2.transAxes, fontsize=40,
# verticalalignment='top')
ax2.yaxis.label.set_size(30)


# 4. Now get gene gradient labels from dendrogram 
method='ward'
metric='euclidean'
Z = linkage(dataframe, method, metric)

ax=None
orientation='top'

R = dendrogram(
    Z=linkage(dataframe, method, metric),
    orientation=orientation,
    get_leaves=True,
    color_threshold=15.0,
    labels=dataframe.index.to_list(),
    distance_sort='ascending',
    above_threshold_color='black', 
    ax=ax, 
    )

gene_list = R['ivl']
label_list = R['leaves_color_list']

gene_gradient_list = pd.DataFrame(
    {'gene_names': gene_list,
     'color_label': label_list,
    })

# Savel gene gradient labels
mapping = {'b': 1, 'r': 2, 'y': 3}
gene_gradient_list = gene_gradient_list.replace({'color_label': mapping})

#gene_gradient_list.to_csv(project_dir+'leana_folder/data/gradient_labels.csv')  



#%% Getting average .nii files for each subject

gene_gradient_list = pd.read_csv(project_dir+'leana_folder/data/gradient_labels.csv', index_col=0)  


#%% Getting average .nii files for each subject

# seperate script: make voxel-wise parcellation 

sub_list = ['15697'] # '9861' done as test already + saved '10021', '12876', '14380', '15496', 

atlas_image = project_dir+'leana_folder/data/cerebellum_voxel_parc.nii'
atlas = atlas_image
#atlas_info = atlas_info_df - need this?


for sub in sub_list: 
    print('\n Starting donor ' +str(sub)+  ' \n')

    # 1. get abagen output for each subject with 'voxel-wise' parcellation 
    sub_df = abagen.get_expression_data(atlas, 
                                            atlas_info=None,
                                            data_dir = ahba_dir,
                                            donors=sub,
                                            ibf_threshold=0.5,
                                            probe_selection='max_intensity',
                                            lr_mirror=False,
                                            exact=True,
                                            tolerance=2,
                                            sample_norm='zscore',
                                            gene_norm='zscore',
                                            region_agg='samples',
                                            agg_metric='mean',
                                            corrected_mni=True,
                                            reannotated=True,
                                            verbose=0) 
    ## Notes
    # started at 3:57pm 
    # finished sometime before 5:30pm 
    
    #sub_df.fillna(0).astype(bool).sum(axis=0)
    # 96 voxels / probes for sub 9861
    
    # 2. get average gene expression within a gradient 
    # voxel id is index value in df above 
    gradient_dict = {1: 'gradient1', 2: 'gradient2', 3: 'gradient3'} 
    
    # initiate new df voxel x gradient average (so 3 columns)
    df_grad_av = pd.DataFrame(index=sub_df.index)
    df_grad_av["gradient1"] = 0.0
    df_grad_av["gradient2"] = 0.0
    df_grad_av["gradient3"] = 0.0
    
    for key, value in gradient_dict.items():
        grad_list = gene_gradient_list.loc[gene_gradient_list['color_label'] == key, 'gene_names']
        grad_list_names = grad_list.tolist()
        #
        grad_list_df = sub_df[[c for c in sub_df.columns if c in grad_list_names]]
        df_grad_av[value] = grad_list_df.mean(axis=1)
        
    # delete rows with nans / no probe (which is most voxels)
    test = df_grad_av.fillna(0).astype(bool).sum(axis=0)
    print('\n *** Number of probes in donor ' +str(sub)+ ' = ' +str(test[0]) + ' *** \n')
    df_grad_av = df_grad_av.dropna(axis='index')
    
    # save gradient averages for each subject 
    df_grad_av.to_csv(project_dir+'leana_folder/data/gradient_average_'+sub+'.csv')  
    print('\n Donor ' +str(sub)+ ' gradient averages saved \n')



#%% Averageing across subjects
# For a single gradient, get .nii files across 6 subjects 
# Best method for averaging? 
# based on Maedbh's approach, just take mean and then normalize 

# 3. Using average values for voxel / probe, project values back into volume / .nii file 
# can do this in pandas first, then project average to single .nii file for each gradient
sub_list = ['9861', '10021', '12876', '14380', '15496', '15697'] 
sub = sub_list[0]

# df_voxels = pd.DataFrame(columns=sub_list) 
# for sub in sub_list: 
#     df_grad_av = pd.read_csv(project_dir+'leana_folder/data/gradient_average_'+sub+'.csv', index_col=0)
#     voxel_inds = df_grad_av.index.values.tolist()
#     df_voxels[sub] = pd.Series(voxel_inds)

# # 411 total voxels across 6 subjects, only 1-3 voxels have data across multiple subs 
# unique_voxels = pd.unique(df_voxels[sub_list].values.ravel('K'))  
# unique_voxels = unique_voxels[~np.isnan(unique_voxels)]
# unique_voxels = unique_voxels.astype(int)

# a. combine merge all sub df's 
sub_dfs = []
for sub in sub_list: 
    df_grad_av = pd.read_csv(project_dir+'leana_folder/data/gradient_average_'+sub+'.csv', index_col=0)
    sub_dfs.append(df_grad_av)
df_allsubs = pd.concat(sub_dfs, axis=0) 

# find duplicate voxels 
duplicated_indexes = df_allsubs.index.duplicated(keep=False)
duplicated_voxels = df_allsubs[duplicated_indexes]
# 4 voxels have probes from two donors 

# b. take average of the 4 overlapping voxels
# 419 total voxels  
df_allsubs = df_allsubs.reset_index(names=['voxel_ids'])
# 415 unique voxels 
df_allsubs = df_allsubs.groupby('voxel_ids').mean().reset_index()
df_allsubs = df_allsubs.sort_values('voxel_ids')


#%% Get .nii gradient files 

probe_list = df_allsubs['voxel_ids'].to_list()

# Fill 3 .nii files w/ values from 3 gradients & save
# load template nii file
atlas_image = project_dir+'leana_folder/data/cerebellum_voxel_parc.nii'
ni = image.load_img((atlas_image))
ni_data = ni.get_fdata()

# Check number of non-zero values 
nonzero = ni_data > 0 
sum_nonzero = sum(sum(sum(nonzero)))

# get list of indices for each non-zero element 
inds = np.where(ni_data > 0)
x_inds = inds[0]
y_inds = inds[1]
z_inds = inds[2]

# initiate new empty 3d array that will have only gene expression values 
g1_vol = np.zeros((256, 256, 256))
g2_vol = np.zeros((256, 256, 256))
g3_vol = np.zeros((256, 256, 256))

# count = 0
# loop through each voxel and replace with a zero or expression value 
for i in range(sum_nonzero):
    x = x_inds[i]
    y = y_inds[i]
    z = z_inds[i]
    voxel_id = int(ni_data[x, y, z])
    #print('voxel id = '+ str(voxel_id))
    if voxel_id in probe_list:
        #count = count+1
        #print('voxel id is in probe list')
        # get expr values for each gradients
        g1_val = df_allsubs.loc[df_allsubs['voxel_ids'] == voxel_id, 'gradient1']
        g2_val = df_allsubs.loc[df_allsubs['voxel_ids'] == voxel_id, 'gradient2']
        g3_val = df_allsubs.loc[df_allsubs['voxel_ids'] == voxel_id, 'gradient3']
        # add data to .nii vol
        g1_vol[x, y, z] = g1_val
        g2_vol[x, y, z] = g2_val
        g3_vol[x, y, z] = g3_val

        
# Check number of non-zero values = 415
nonzero_g = abs(g1_vol) > 0
sum_g = sum(sum(sum(nonzero_g)))
print(sum_g)


# Save .nii files 
g1_img = image.new_img_like(ni, g1_vol, copy_header=True)
g1_img.to_filename(project_dir+'leana_folder/data/gradient_1.nii')

g2_img = image.new_img_like(ni, g2_vol, copy_header=True)
g2_img.to_filename(project_dir+'leana_folder/data/gradient_2.nii')

g3_img = image.new_img_like(ni, g3_vol, copy_header=True)
g3_img.to_filename(project_dir+'leana_folder/data/gradient_3.nii')









