#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:02:53 2023

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
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures 


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

    # sets the line weight for border of graph
    # for axis in ['top','bottom','left','right']:
    #     ax2.spines[axis].set_linewidth(4)

#%%
ahba_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid/AHBA/'
sub_dir = '/Users/leanaking/freesurfer/subjects/AHBA/'
project_dir = '/Users/leanaking/Projects/cerebellum_transcriptomics/'

atlas_dir = project_dir + '/external/atlas_templates/'
fig_dir = project_dir + 'leana_folder/figures/'



#%% Replicate Supp Figure 2 plot

# plt.figure(num=1, figsize=[10,15])
# test = figures.supp_2(atlas1="MDTB-10-subRegions", atlas2="SUIT-10", which_genes=which_genes, 
#                atlas_other="MDTB-10", percentile=percentile, classifier='logistic', 
#                remove_outliers=True, normalize=True)
# plt.savefig(fig_dir+'Supplemental_BS/test'+str(percentile)+'.png')
# plt.clf()


percentile = 1
which_genes='top_bs'
atlas1="MDTB-10-subRegions"
atlas2="SUIT-10"
atlas_other="MDTB-10"
normalize = True
classifier='logistic'


# supp 2a
df = ana.return_thresholded_data(atlas=atlas1, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=False, normalize=normalize, all_samples=True)
plt.figure()
visualize.confusion_matrix_plot(atlas1, df, classifier=classifier, label_type="binary") # ax=ax1
plt.tight_layout()
plt.savefig(fig_dir+'Supplemental_BS/2A_per'+str(percentile)+'.png', bbox_inches="tight", dpi=600)
#plt.text(0, 0, 'A', fontsize=40, verticalalignment='top')


# get F1 score from plot above
df = ana.return_thresholded_data(atlas=atlas1, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=False, normalize=normalize, all_samples=True)
X, y, classes = ana._get_X_y(atlas1, df, label_type="binary")
cm, f1 = ana._confusion_matrix(X, y, classifier=classifier, label_type="binary")
print(f'f1 = {f1}')

# supp 2b
df = ana.return_thresholded_data(atlas=atlas2, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=False, normalize=normalize, all_samples=True)
plt.figure()
visualize.confusion_matrix_plot(atlas2, df, classifier=classifier, label_type="multi-class") # ax=ax1
plt.tight_layout()
plt.savefig(fig_dir+'Supplemental_BS/2B_per'+str(percentile)+'.png', bbox_inches="tight", dpi=600)



#%% Replicate Supp Figure 3 plot

atlas="SUIT-10"
which_genes='top_bs'
percentile=1
remove_outliers=True
atlas_other="MDTB-10" 
normalize=True

plotting_style()

fig = plt.figure(figsize=(15,15))

gs = GridSpec(6, 3, figure=fig)
x_pos = -0.18
y_pos = 1.1

for i in np.arange(0,6):
    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, donor_num=i+1, remove_outliers=remove_outliers, normalize=normalize)

    ax1 = fig.add_subplot(gs[i, 0])
    if i==0:
        ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=50,
        verticalalignment='top')
    visualize.pcs_loading_plot(df, pcs=[1], group_pcs=False, ax=ax1, atlas=atlas)
    ax1.tick_params(axis='x', which='major', labelsize=30, rotation=35)
    ax1.tick_params(axis='y', which='major', labelsize=30)
    
    ax1.set_title(str(Defaults.donors[i]), fontsize=40)
    ax1.yaxis.label.set_size(40)

    ax2 = fig.add_subplot(gs[i, 1])
    if i==0:
        ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=50,
        verticalalignment='top')
    visualize.pcs_loading_plot(df, pcs=[2], group_pcs=False, ax=ax2, atlas=atlas)
    ax2.tick_params(axis='x', which='major', labelsize=30, rotation=35)
    ax2.tick_params(axis='y', which='major', labelsize=30)
    ax2.yaxis.label.set_size(40)

    ax3 = fig.add_subplot(gs[i, 2])
    if i==0:
        ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=50,
        verticalalignment='top')
    visualize.dendrogram_plot(df.T, ax=ax3, color_leaves=False)
    ax3.tick_params(axis='x', which='major', labelsize=30, rotation=35)
    ax3.tick_params(axis='y', which='major', labelsize=30)
    ax3.yaxis.label.set_size(0)

plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.5, hspace=1.0)
plt.savefig(fig_dir+'Supplemental_BS/3_per'+str(percentile)+'.png', bbox_inches="tight", dpi=600)


## To get code to run, the following adjustment was made in visualize.py: 
    # def _make_colorpalette(atlas):

    # cm, labels = _make_colormap(atlas)

    # #mpl.cm.register_cmap("mycolormap", cm) - this line was commented out - line 329
    # cpal = sns.color_palette("mycolormap", n_colors=len(labels))



#%% Replicate Supp Figure 4 plot

atlas="SUIT-26"
which_genes='top_bs'
percentile=1
remove_outliers=True
atlas_other="MDTB-10"
normalize=True
    
plt.clf()
plotting_style()

fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
dataframe = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
visualize.simple_corr_heatmap(dataframe, atlas=atlas, distance_correct=True, ax=ax1)
plt.savefig(fig_dir+'Supplemental_BS/4_per'+str(percentile)+'.png', bbox_inches="tight", dpi=600)




