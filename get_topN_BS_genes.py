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
from transcriptomics import gec_functions_preprocess as preproc 

atlas = 'SUIT-10' #"MDTB-10-subRegions" #'MDTB-10-subRegions' # will need to do for all atlases
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
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures 

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
        color_threshold=65.0,  # 1% = 35, 5% = 45, 
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
plt.savefig(fig_dir+'Fig2_BS/B_dendo_'+atlas+'_per'+str(percentile)+'_BS.png')




#%% Replicate rest of Figure 2 plots 

## Just learned that figures C-E do not change w/ percentile in method1 (whole plot) above
# will need to plot each sub-fig seperately as well
percentiles = [1] # 5, 10, 25, 50]
#fig, (ax1, ax2, ax3) = plt.subplots(3)
for percentile in percentiles: 
    df = ana.return_grouped_data(atlas='MDTB-10-subRegions', which_genes='top', atlas_other=atlas_other, percentile=percentile, reorder_labels=True, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas='MDTB-10-subRegions', distance_correct=True)#, ax=ax1)
    plt.savefig(fig_dir+'Fig2_BS/C_heatmap_per'+str(percentile)+'.png')
    plt.clf()

    # 2D
    atlas_other="MDTB-10"
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-7", atlas_cortex="Yeo-7", which_genes='top', atlas_other=atlas_other, percentile=percentile, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-7", distance_correct=True, simple_labels=True)# ax = ax2)
    plt.savefig(fig_dir+'Fig2_BS/D_heatmap_per'+str(percentile)+'.png')
    plt.clf()
    
    # 2E
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-17", atlas_cortex="Yeo-17", which_genes='top', atlas_other=atlas_other, percentile=percentile, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-17", distance_correct=True, simple_labels=True)#, ax = ax3)
    plt.savefig(fig_dir+'Fig2_BS/E_heatmap_per'+str(percentile)+'.png')
    plt.clf()

    

#%% Replicate Figure 3 plots

## Get percentile df for input
percentile = 1
atlas='SUIT-10'
which_genes='top_bs'
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
plt.savefig(fig_dir+'Fig3_BS/B_dendo_'+atlas+'_per'+str(percentile)+'_BS.png')



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
plt.savefig(fig_dir+'Fig3_BS/B_raster_'+atlas+'_per'+str(percentile)+'_BS.png')



# 3c
ax = visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.savefig(fig_dir+'Fig3_BS/C_corrmat_'+atlas+'_per'+str(percentile)+'_BS.png')




# 3d
plt.figure(num=1, figsize=[8,6])
visualize.dendrogram_plot(df.T, color_leaves=False)
plt.tick_params(axis='x', which='major', labelsize=15, rotation=45)
plt.tick_params(axis='y', which='major', labelsize=25)
plt.tight_layout()
#plt.yaxis.label.set_size(20)
plt.savefig(fig_dir+'Fig3_BS/D_dendo_'+atlas+'_per'+str(percentile)+'_BS.png')




# 3e
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
plt.savefig(fig_dir+'Fig3_BS/E_pcBar_'+atlas+'_per'+str(percentile)+'_BS.png')





#%% Replicate Figure 4 plots

image1="rat_zones"
image2="MDTB-10-subRegions-transcriptomic"
atlas='MDTB-10-subRegions' 
which_genes='top_bs' 
percentile=1
remove_outliers=True 
atlas_other="MDTB-10"
normalize=True


# 4b - this just loads the original image
visualize.png_plot(filename=image2)
# Will need to make png file seperately



### Make -info.csv file for rgb values per atlas ROI - works 
atlas="MDTB-10-subRegions"
atlas_other="MDTB-10"
remove_outliers=True
normalize=True
import re
import os

df = ana.return_grouped_data(atlas=atlas, atlas_other=atlas_other,  which_genes='top_bs', remove_outliers=remove_outliers, normalize=normalize)

R = visualize.dendrogram_plot(df.T)

regex = r"(\d+)-(\w+)"

# get atlas labels
groups = []
for p in R['ivl']:
    match = re.findall(regex, p)[0]
    groups.append(match)

# get indices for labels
index = []
for group in groups:
    if group[1]=='A':
        index.append(int(group[0]))
    else:
        index.append(int(group[0]) + 10)

# zero index the regions
index = [i-1 for i in index] # zero index

# figure out which regions are missing
res = [ele for ele in range(max(index)+1) if ele not in index]

for reorder in [True, False]:
    # assign colors to clusters
    colors = sns.color_palette("coolwarm", len(index))
    # convert to list
    colors = [list(ele) for ele in colors]
    if reorder:
        # append NaN color values to missing regions
        for ele in res:
            colors.append(np.tile(float("NaN"),3).tolist())
        # put the rgb colors in sorted order
        colors_dendrogram = [x[1] for x in sorted(zip(index+res, colors), key=lambda x: x[0])]
        labels = Defaults.labels[atlas] 
        outname = f"{atlas}-transcriptomic-BS_TEST-info.csv"
    else:
        # don't sort the rgb colors
        colors_dendrogram = colors[::-1]
        labels = R['ivl'][::-1]
        outname = f"{atlas}-transcriptomic-dendrogram-ordering-BS_TEST-info.csv"

    color_r = []
    color_g = []
    color_b = []
    for i in np.arange(len(colors_dendrogram)):
        color_r.append(np.round(colors_dendrogram[i][0],2))
        color_g.append(np.round(colors_dendrogram[i][1],2))
        color_b.append(np.round(colors_dendrogram[i][2],2))

    data = {'region_num':list(range(1,len(labels)+1)), 'region_id': labels, 'r': color_r, 'g':color_g, 'b':color_b}

    # create dataframe
    df_new = pd.DataFrame(data) 

    df_new.to_csv(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", outname))






#### Using info.csv file, now make png - this where I'm stuck ####################
from nilearn import plotting, datasets, surface
from PIL import Image

atlas_cm = "MDTB-10-subRegions-transcriptomic-dendrogram-ordering-BS_TEST"
cm, labels = visualize._make_colormap(atlas=atlas_cm)


atlas='MDTB-10-subRegions' 
info_file=True
mesh='very_inflated'
hemisphere="L"
view="lateral"
ax=None
save=True
resize=True

# get mesh
#surf_mesh = os.path.join(Defaults.SURF_DIR, f"fsaverage.{hemisphere}.{mesh}.32k_fs_LR.surf.gii")
surf_mesh = os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", 'FLAT.surf.gii') 
test_mesh = surface.load_surf_mesh(str(surf_mesh))

# get surface
## do I need to make a new label.gii file w/ new colorbar values? - nope, this is the region number + doesn't need to change
surf_map = 'MDTB-10-subRegions.label.gii'
surf_data = surface.load_surf_data(str(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", surf_map)))
# has nan's - get rid of those?
surf_data[np.isnan(surf_data)] = 0


# get colors
atlas_cm = "MDTB-10-subRegions-transcriptomic-dendrogram-ordering-BS_TEST"
cm, labels = visualize._make_colormap(atlas=atlas_cm)

# name of output file 
filename = Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{atlas}-{hemisphere}-{view}-BS_test.png"


# # make plot 
# hem = "left"
# # making the plot!
# plotting.plot_surf_roi(test_mesh, roi_map=surf_data, hemi=hem, view=view, 
#             cmap=cm, darkness=0.5, output_file=str(filename))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plotting.plot_surf_roi(str(surf_mesh), roi_map=surf_data,
#         hemi=hem, view=view, axes=ax, cmap=cm, darkness=.5)
# plotting.show()



## This works!!! - but colormap is off maybe? 
# to check this and make the colorbar w/ the ROI labels 
view = plotting.view_surf(str(surf_mesh), surf_map=surf_data, cmap=cm)
view.open_in_browser()


# visualize.png_plot
## concat_png
### _make_all_png
#### _make_png_cortex
#####






