#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:37:28 2024

Replicate Figure 4B plot with sensitivity approach 

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
from scipy.cluster.hierarchy import dendrogram, linkage
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

print(Defaults.PROCESSED_DIR)

#%% Make -info.csv file for rgb values per atlas ROI - works 

## Done for bs: 1, 5, 10, and original, 1, 5, 10, 25, 50

###
percentile = 10
which_genes = 'top'
cond = 'top10'  
#which_genes = 'top_bs'
#cond = 'BS-top1'
###


# defaults 
normalize=True
remove_outliers=True 
atlas="MDTB-10-subRegions"
atlas_other="MDTB-10"
remove_outliers=True
normalize=True

# get df
df = ana.return_grouped_data(atlas=atlas, atlas_other=atlas_other, percentile=percentile, 
                             which_genes=which_genes, remove_outliers=remove_outliers, 
                             normalize=normalize)

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

for reorder in [True, False]:  #[True, False]
    # assign colors to clusters
    colors = sns.color_palette("coolwarm", len(index))
    #greys = sns.color_palette("Greys", 1)
    # convert to list
    colors = [list(ele) for ele in colors]
    #nan_colors = [list(res) for ele in greys]
    if reorder:
        # append NaN color values to missing regions
        for ele in res:
            colors.append(np.tile(float("NaN"),3).tolist())
        # put the rgb colors in sorted order
        colors_dendrogram = [x[1] for x in sorted(zip(index+res, colors), key=lambda x: x[0])]
        labels = ['R01-A', 'R02-A', 'R03-A', 'R04-A', 'R05-A', 'R06-A', 'R07-A', 'R08-A', 'R09-A', 'R10-A', 'R01-P', 'R02-P', 'R03-P', 'R04-P', 'R05-P', 'R06-P', 'R07-P', 'R08-P', 'R09-P', 'R10-P']
        outname = f"{atlas}-transcriptomic-{cond}-info.csv"
    else:
        # don't sort the rgb colors
        colors_dendrogram = colors[::-1]
        labels = R['ivl'][::-1]
        outname = f"{atlas}-transcriptomic-dendrogram-ordering-{cond}-info.csv"

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


#%% New edit^ adjust colorbar to match dendrogram 

## bs: 1, 5, 10, 
## original: 1, 5, 10, 25, 50

## Done for bs: 1, 5, 10
## and original: 1, 5, 10, 25
    
###
percentile = 1
which_genes = 'top'
cond = 'top1'  
#which_genes = 'top_bs'
#cond = 'BS-top1'
###

# defaults 
normalize=True
remove_outliers=True 
atlas="MDTB-10-subRegions"
atlas_other="MDTB-10"
remove_outliers=True
normalize=True

# get df
df = ana.return_grouped_data(atlas=atlas, atlas_other=atlas_other, percentile=percentile, 
                             which_genes=which_genes, remove_outliers=remove_outliers, 
                             normalize=normalize)
method='ward' 
metric='euclidean' 
reorder=True 
orientation='top' 
color_leaves=True 
ax=None

hierarchy.set_link_color_palette(['b', 'r', 'k'])


####
R = dendrogram(
        Z=linkage(df.T, method, metric),
        orientation=orientation,
        color_threshold=250.0,  # 35, 75, 100, 175, 250 ##################
        get_leaves=True,
        labels=df.T.index.to_list(),
        distance_sort='ascending',
        above_threshold_color='black',
        leaf_rotation=45)
####


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
res = [ele for ele in range(max(index)+1) if ele not in index]

# get clust order info as df 
clust_tuple = tuple(zip(R['ivl'], R['leaves_color_list'], index))
clust_df = pd.DataFrame(clust_tuple, columns=['Region', 'color', 'index'])

# Get sub-color based of index 
# 17 regions, 8 posterior, 9 anterior 
    # but 5 blue and 12 red (for top1 BS)- this will vary 

# 1. get colors based on number in cluster
num_blue = clust_df['color'].value_counts()['b']
num_red = clust_df['color'].value_counts()['r']
blues = sns.color_palette("Blues", num_blue)
blues = [list(ele) for ele in blues]
reds = sns.color_palette("Reds", num_red)
reds = [list(ele) for ele in reds]

colors = blues + reds

# 2. Get indices for each color 
inds_blue = clust_df.loc[clust_df['color'] == 'b', 'index'].to_numpy()
inds_red = clust_df.loc[clust_df['color'] == 'r', 'index'].to_numpy()
## might not need step 2 + 3
# 3. sort sub-colors by index for each color
colors_blues = [x[1] for x in sorted(zip(inds_blue, blues), key=lambda x: x[0])]
colors_reds = [x[1] for x in sorted(zip(inds_red, reds), key=lambda x: x[0])]


for ele in res:
    colors.append(np.tile(float("NaN"),3).tolist())
# put the rgb colors in sorted order
colors_sorted = [x[1] for x in sorted(zip(index+res, colors), key=lambda x: x[0])]
labels_sorted = ['R01-A', 'R02-A', 'R03-A', 'R04-A', 'R05-A', 'R06-A', 'R07-A', 'R08-A', 'R09-A', 'R10-A', 'R01-P', 'R02-P', 'R03-P', 'R04-P', 'R05-P', 'R06-P', 'R07-P', 'R08-P', 'R09-P', 'R10-P']

#
color_r = []
color_g = []
color_b = []
for i in np.arange(len(colors_sorted)):
    color_r.append(np.round(colors_sorted[i][0],2))
    color_g.append(np.round(colors_sorted[i][1],2))
    color_b.append(np.round(colors_sorted[i][2],2))

data = {'region_num':list(range(1,len(labels_sorted)+1)), 'region_id': labels_sorted, 'r': color_r, 'g':color_g, 'b':color_b}

# create dataframe
df_cb = pd.DataFrame(data) 

outname = f"{atlas}-transcriptomic-lk_custom-{cond}-info.csv"
df_cb.to_csv(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", outname))


#%% Using info.csv file, make colorbar
from nilearn import plotting, datasets, surface
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

#percentile = 1
#which_genes = 'top' #'top_bs'
#cond = 'top1' #'BS-top10'

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
        
# Adding white color to front for nans in surf file 
labels.insert(0, 'nan')               
colors_all.insert(0, [0.99, 0.99, 0.99])        
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
cb3.ax.tick_params(axis='y', which='major', labelsize=28)
plt.savefig(str(Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{atlas}-colorbar-lk_custom-{cond}.png"), bbox_inches='tight')


#%% Make plotting version^ of colorbar

percentile = 1
which_genes = 'top_bs' #'top_bs' or 'top'
cond = 'BS-top1' #'BS-top10' or 'top10'

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
plt.savefig(str(Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{atlas}-colorbar-lk_custom-PLOT-{cond}.png"), bbox_inches='tight')


# Get rotated text
ax=None
if ax is None:
    fig, ax = plt.subplots(figsize=(14, 1))
bounds = np.arange(cmap.N)# + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, 
                                norm=norm,
                                ticks=bounds,
                                format='%s',
                                orientation='horizontal',
                                )
cb3.set_ticklabels(labels)  #labels[::-1]
cb3.ax.tick_params(axis='x', which='major', labelsize=38, rotation=45)
plt.savefig(fig_dir+'/Fig4/rotated_axis_text.png', dpi=500, 
           bbox_inches='tight', transparent=True)

#%% Get topoplot 

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
view = plotting.view_surf(str(surf_mesh), surf_map=surf_data, cmap=cmap, 
                          symmetric_cmap=False)
view.open_in_browser()
#view.savefig(fig_dir+'test', bbox_inches="tight", dpi=600)


# ### Trying to get a png file to save 
# from matplotlib import pyplot
# filename = fig_dir+'test'
# # Try to get png plot - much easier for saving 
# plotting.plot_surf(str(surf_mesh), roi_map=surf_data,
#                 hemi='left', view='lateral', cmap=cm, darkness=.5,
#                 output_file=str(filename)) 
# pyplot.show()
# plotting.show()


# plotting.plot_surf(str(surf_mesh), surf_map=surf_data, cmap=cmap, symmetric_cmap=False)
# plt.gcf().savefig(fig_dir+'test', bbox_inches="tight")
# plotting.view_surf(str(surf_mesh)).open_in_browser()
# plt.close("all")



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# view = plotting.plot_surf(str(surf_mesh), surf_map=surf_data,
#        hemi='left', view='lateral', cmap=cm, darkness=.5, axes=ax)
# view.open_in_browser()

# pyplot.show()

# plotting.show()




#%% Make labels for dice coeff calculation

labels = ['R01-A', 'R02-A', 'R03-A', 'R04-A', 'R05-A', 'R06-A', 'R07-A', 'R08-A', 'R10-A', 'R01-P', 'R03-P', 'R04-P', 'R05-P', 'R06-P', 'R07-P', 'R08-P', 'R10-P'] # R09-A, R02-P, and R09-P missing
GT_labels = [0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1]
GT_ind = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17, 19]
GT_tup_label = tuple(zip(labels, GT_labels))
GT_tup_index = tuple(zip(labels, GT_ind))

GT_clust_df = pd.DataFrame(GT_tup_label, columns=['Region', 'Label'])
GT_ind_df = pd.DataFrame(GT_tup_index, columns=['Region', 'Label'])

###
#percentile = 10
#which_genes = 'top_bs'
#cond = 'BS-top10'
###

# defaults 
normalize=True
remove_outliers=True 
atlas="MDTB-10-subRegions"
atlas_other="MDTB-10"
remove_outliers=True
normalize=True

# get df
df = ana.return_grouped_data(atlas=atlas, atlas_other=atlas_other, percentile=percentile, 
                             which_genes=which_genes, remove_outliers=remove_outliers, 
                             normalize=normalize)
method='ward' 
metric='euclidean' 
reorder=True 
orientation='top' 
color_leaves=True 
ax=None

R = dendrogram(
        Z=linkage(df.T, method, metric),
        orientation=orientation,
        color_threshold=100.0,
        get_leaves=True,
        labels=df.T.index.to_list(),
        distance_sort='ascending',
        above_threshold_color='black')


binary_color = []
for color in R['leaves_color_list']:
    if color == 'b':
        binary_color.append(0)
    elif color == 'r':
        binary_color.append(1)
    else:
        print('more than one color present')

clust_tuple = tuple(zip(R['ivl'], binary_color))

index.reverse()
ind_tuple = tuple(zip(R['ivl'], index))

clust_df = pd.DataFrame(clust_tuple, columns=['Region', 'Label'])
clust_df = clust_df.set_index('Region')
clust_df = clust_df.reindex(index=GT_clust_df['Region'])
clust_df = clust_df.reset_index()

ind_df = pd.DataFrame(ind_tuple, columns=['Region', 'Label'])
ind_df = ind_df.set_index('Region')
ind_df = ind_df.reindex(index=GT_ind_df['Region'])
ind_df = ind_df.reset_index()


## For now, use binary cluster labels for dice coeff
# seems like this is a binary measure
u = GT_clust_df[["Label"]].to_numpy().flatten()
v = clust_df[["Label"]].to_numpy().flatten()
coef = distance.dice(u, v)

print(cond)
print(coef)



#%% Resize topoplot
# doesn't give the right shape, will need to adjust this in illustrator

image = Image.open(str(fig_dir+ 'Fig4_BS/top1_BS.png'))
atlas_size = image.size
newsize = (int(atlas_size[0]/4), int(atlas_size[1]/4)) 
image = image.resize(newsize)
image.save(str(fig_dir+ 'Fig4_BS/top1_BS_resized.png'))


