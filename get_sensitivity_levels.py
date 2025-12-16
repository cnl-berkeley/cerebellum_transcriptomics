#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:01:12 2024

@author: leanaking
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
import re
import os
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from scipy.cluster import hierarchy
from scipy.spatial import distance

#%%
# Using maedbh's package
# not really sure how to do the class thing yet
import transcriptomics
from transcriptomics import data #process_gene_expression
from transcriptomics.constants import Defaults
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 


#%%
ahba_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid/AHBA/'
sub_dir = '/Users/leanaking/freesurfer/subjects/AHBA/'
project_dir = '/Users/leanaking/Projects/cerebellum_transcriptomics/'

atlas_dir = project_dir + '/external/atlas_templates/'
fig_dir = project_dir + 'leana_folder/figures/'



#%% Replicate Figure 2B plot 
## Get percentile df for input
percentile = 50
atlas='MDTB-10-subRegions'
atlas_other = "MDTB-10" #"MDTB-10-subRegions"
df = ana.return_grouped_data(atlas=atlas, which_genes='top', percentile=percentile, atlas_other=atlas_other,
                             remove_outliers=True, normalize=True)
df = df.set_index(df.index)

# get dataframe input
dataframe = df.T
dataframe = visualize._reorder_dendrogram_leaves(dataframe)

# Plot dendrogram seperately 
set_link_color_palette(['b', 'r', 'y', 'm'])
plt.figure(num=1, figsize=[30,10])
with plt.rc_context({'lines.linewidth': 3}):
    R = dendrogram(
            Z=linkage(dataframe, method='ward', metric='euclidean'),
            orientation='top',
            color_threshold=250.0,  # 1% = 35, 5% = 75, 10% = 120, 25%=175, 50%=250 
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
#plt.rcParams['lines.linewidth'] = 3
plt.tight_layout()
#
plt.savefig(fig_dir+'Fig2/B_PLOT_dendo_'+atlas+'_per'+str(percentile)+'_2clsut.png', bbox_inches="tight", dpi=600)


#%% Using info.csv file, make colorbar for topoplots
from nilearn import plotting, datasets, surface
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

percentile = 50
which_genes = 'top' #'top_bs'
cond = 'top50' #'BS-top10'

# defaults 
atlas="MDTB-10-subRegions"

# Load info file
#atlas_cm = 'MDTB-10-subRegions-transcriptomic-dendrogram-ordering-'+cond 
#atlas_cm = 'MDTB-10-subRegions-transcriptomic-'+cond 
atlas_cm = 'MDTB-10-subRegions-transcriptomic-lk_custom-'+cond 


# Try _make_colormap manually: 
atlas_info = Defaults.EXTERNAL_DIR / "atlas_templates" / f"{atlas_cm}-info.csv"
df = pd.read_csv(atlas_info, index_col=0)
colors = df.apply(lambda x: list([x['r'],
                                x['g'],
                                x['b']]), axis=1) 

# Make sure these two variables have all rois and values (including nans) present
regions = df['region_id']
colors_rgb = colors

# add grey to nan regions 
colors_all = []
labels = []
for i, rgb in enumerate(colors_rgb):
    if any([np.isnan(x) for x in rgb]):
        colors_all.append([0.45, 0.45, 0.45])
        labels.append(regions[i])
        print('test')
    else:
        print(i)
        colors_all.append(rgb)
        labels.append(regions[i])
        
# # Adding white color to front for nans in surf file 
labels.insert(0, 'nan')               
colors_all.insert(0, [0.99, 0.99, 0.99])        
cm = LinearSegmentedColormap.from_list(atlas, colors_all, N=len(colors_all))
cmap_topo = cm



# Save colorbar (w/ corresponding ROI) using _make_colorbar
ax=None
if ax is None:
    fig, ax = plt.subplots(figsize=(1, 14))
bounds = np.arange(cmap_topo.N)# + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap_topo.N)
cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap_topo.reversed(cmap_topo), 
                                norm=norm,
                                ticks=bounds,
                                format='%s',
                                orientation='vertical',
                                )
cb3.set_ticklabels(labels[::-1])  #labels[::-1]
cb3.ax.tick_params(axis='y', which='major', labelsize=28)
#plt.savefig(str(Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{atlas}-colorbar-lk_custom-{cond}.png"), bbox_inches='tight')


#%% Make plotting version^ of colorbar


# Try _make_colormap manually: 
atlas_info = Defaults.EXTERNAL_DIR / "atlas_templates" / f"{atlas_cm}-info.csv"
df = pd.read_csv(atlas_info, index_col=0)
colors = df.apply(lambda x: list([x['r'],
                                x['g'],
                                x['b']]), axis=1) 

# Make sure these two variables have all rois and values (including nans) present
regions = df['region_id']
colors_rgb = colors

# add grey to nan regions 
colors_all = []
labels = []
for i, rgb in enumerate(colors_rgb):
    if any([np.isnan(x) for x in rgb]):
        print('Empty roi - dont add')
    else:
        print(i)
        colors_all.append(rgb)
        labels.append(regions[i])
          
        
cm = LinearSegmentedColormap.from_list(atlas, colors_all, N=len(colors_all))
cmap = cm

# Save colorbar (w/ corresponding ROI) using _make_colorbar
ax=None
if ax is None:
    fig, ax = plt.subplots(figsize=(1, 14))
bounds = np.arange(cmap.N)# + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap.reversed(cmap), 
                                norm=norm,
                                ticks=bounds,
                                format='%s',
                                orientation='vertical',
                                )
cb3.set_ticklabels(labels[::-1])  #labels[::-1]
cb3.ax.tick_params(axis='y', which='major', labelsize=34)
plt.savefig(fig_dir+'Fig4/colorbar_'+atlas+'_'+str(cond)+'_orig.png', bbox_inches="tight", dpi=600)


#%% Plot 
atlas='MDTB-10-subRegions' 
info_file=True
mesh='very_inflated'
hemisphere="L"
view="lateral"
ax=None
save=True
resize=True

# get mesh
surf_mesh = os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", 'FLAT.surf.gii') 
test_mesh = surface.load_surf_mesh(str(surf_mesh))

# get surface
surf_map = 'MDTB-10-subRegions.label.gii'
surf_data = surface.load_surf_data(str(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", surf_map)))
#surf_data = surf_data-1
# has nan's - get rid of those?
surf_data[np.isnan(surf_data)] = 0


# Plot HTML
view = plotting.view_surf(str(surf_mesh), surf_map=surf_data, cmap=cmap_topo, 
                          symmetric_cmap=False)
view.open_in_browser()
#view.savefig(fig_dir+'test', bbox_inches="tight", dpi=600)
