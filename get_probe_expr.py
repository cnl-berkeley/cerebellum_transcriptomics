#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:18:14 2024

Getting processed expression data per probe 
(not grouped into ROIs)

@author: leanaking
"""
import abagen
import numpy as np
import pandas as pd
from nilearn import image

#%%
# Using maedbh's package
import transcriptomics
from transcriptomics import data #process_gene_expression
from transcriptomics.constants import Defaults

from transcriptomics import gec_functions_preprocess as preproc 
from transcriptomics import gec_functions_ana as ana 
from transcriptomics import visualize 
from transcriptomics import figures 

# abagen parameters
Parameters = preproc.Parameters
   
#atlas = 'MDTB-10'  
#atlas_files = preproc._get_atlas_files(atlas)
#atlas_info = pd.read_csv(atlas_files.info)


#%%
ahba_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid/AHBA/'
sub_dir = '/Users/leanaking/freesurfer/subjects/AHBA/'
project_dir = '/Users/leanaking/Projects/cerebellum_transcriptomics/'

atlas_dir = project_dir + 'external/atlas_templates/'
fig_dir = project_dir + 'leana_folder/figures/'
save_dir = project_dir + 'leana_folder/data/'


#%%
### Load parcellation image and data to use for mask 
cortical_parcellations = ['Yeo-7', 'Yeo-17', 'Desikan-Killiany-83']

cerebellar_parcellations = ['MDTB-10', 'MDTB-10-subRegions', 'SUIT-10', 'SUIT-26',
                            'Buckner-7', 'Buckner-17', 'Ji-10']

#%%
# create cerebellum mask

for atlas_str in cerebellar_parcellations:

    atlas = atlas_dir+atlas_str+'.nii'
    atlas_img = image.load_img(atlas)
    atlas_data = atlas_img.get_fdata()
    
    mask_data = np.zeros((atlas_data.shape))
    mask_ones = np.where(atlas_data > 0)
    mask_data[mask_ones] = 1
    
    mask = image.new_img_like(atlas, mask_data, copy_header=True)
    
    probe_expr, probe_coords = abagen.get_samples_in_mask(mask=mask,
                               data_dir = ahba_dir,
                               donors='all',
                               ibf_threshold=Parameters.ibf_threshold,
                               probe_selection=Parameters.probe_selection,
                               #donor_probes='independent',
                               sim_threshold=None,
                               lr_mirror=Parameters.lr_mirror,
                               exact=Parameters.exact, missing=None,
                               tolerance=Parameters.tolerance,
                               sample_norm=Parameters.sample_norm,
                               gene_norm=Parameters.gene_norm,
                               norm_matched=True,
                               norm_structures='cerebellum',  #'cortex' or 'cerebellum' #mmm, maybe this should actually be 'False'?
                               region_agg=Parameters.region_agg,
                               agg_metric=Parameters.agg_metric,
                               corrected_mni=True,
                               reannotated=Parameters.reannotated,
                               return_counts=True,
                               return_donors=False,
                               return_report=True,
                               verbose=0) 
    
    # check that 426 probes (from MDTB-10) is the same number of total probes Maebh used
    # interesting - SUIT-10 is 344 probes
    num_probes = len(probe_expr)
    print('\n********** ')
    print('For atlas: ' +atlas_str+ ', number of probes = ' + str(num_probes))
    print('********** \n')

    probe_expr.to_csv(save_dir + 'probe_expr_NC_cerebellum_'+atlas_str+'.csv')  
    probe_coords.to_csv(save_dir + 'probe_coords_NC_cerebellum_'+atlas_str+'.csv')  



#%%
# create cerebellum + cortical mask
atlas_str = 'MDTB-10'
atlas = atlas_dir + '/'+atlas_str+'.nii'
atlas_img = image.load_img(atlas)
atlas_data = atlas_img.get_fdata()
mask_data = np.zeros((atlas_data.shape))
mask_ones_cerebellum = np.where(atlas_data > 0)

atlas_str = 'Yeo-17'
atlas = atlas_dir + '/'+atlas_str+'.nii'
atlas_img = image.load_img(atlas)
atlas_data = atlas_img.get_fdata()
mask_data = np.zeros((atlas_data.shape))
mask_ones_cortex= np.where(atlas_data > 0)

mask_data[mask_ones_cerebellum] = 1
mask_data[mask_ones_cortex] = 1
atlas_str = ''

mask = image.new_img_like(atlas, mask_data, copy_header=True)

probe_expr, probe_coords = abagen.get_samples_in_mask(mask=mask,
                           data_dir = ahba_dir,
                           donors='all',
                           ibf_threshold=Parameters.ibf_threshold,
                           probe_selection=Parameters.probe_selection,
                           #donor_probes='independent',
                           sim_threshold=None,
                           lr_mirror=Parameters.lr_mirror,
                           exact=Parameters.exact, missing=None,
                           tolerance=Parameters.tolerance,
                           sample_norm=Parameters.sample_norm,
                           gene_norm=Parameters.gene_norm,
                           norm_matched=True,
                           norm_structures='cortex',  #'cortex' or 'cerebellum' - post review update: changed this to false
                           region_agg=Parameters.region_agg,
                           agg_metric=Parameters.agg_metric,
                           corrected_mni=Parameters.corrected_mni,
                           reannotated=Parameters.reannotated,
                           return_counts=True,
                           return_donors=False,
                           return_report=True,
                           verbose=0) 

# check that 426 probes (from MDTB-10) is the same number of total probes Maebh used
# interesting - SUIT-10 is 344 probes
num_probes = len(probe_expr)
print('\n********** ')
print('For atlas: ' +atlas_str+ ', number of probes = ' + str(num_probes))
print('********** \n')

probe_expr.to_csv(save_dir + 'probe_expr_cerebellum_andcortex_cortnorm.csv')  
probe_coords.to_csv(save_dir + 'probe_coords_cerebellum_andcortex_cortnorm.csv')  


#%%
# create cortical mask
atlas_str = 'Yeo-17'
atlas = atlas_dir + '/'+atlas_str+'.nii'
atlas_img = image.load_img(atlas)
atlas_data = atlas_img.get_fdata()
mask_data = np.zeros((atlas_data.shape))
mask_ones_cortex= np.where(atlas_data > 0)

mask_data[mask_ones_cortex] = 1

mask = image.new_img_like(atlas, mask_data, copy_header=True)

probe_expr, probe_coords = abagen.get_samples_in_mask(mask=mask,
                           data_dir = ahba_dir,
                           donors='all',
                           ibf_threshold=Parameters.ibf_threshold,
                           probe_selection=Parameters.probe_selection,
                           #donor_probes='independent',
                           sim_threshold=None,
                           lr_mirror=Parameters.lr_mirror,
                           exact=Parameters.exact, missing=None,
                           tolerance=Parameters.tolerance,
                           sample_norm=Parameters.sample_norm,
                           gene_norm=Parameters.gene_norm,
                           norm_matched=True,
                           norm_structures='cerebellum',  #'cortex' or 'cerebellum'
                           region_agg=Parameters.region_agg,
                           agg_metric=Parameters.agg_metric,
                           corrected_mni=Parameters.corrected_mni,
                           reannotated=Parameters.reannotated,
                           return_counts=True,
                           return_donors=False,
                           return_report=True,
                           verbose=0) 

# check that 426 probes (from MDTB-10) is the same number of total probes Maebh used
# interesting - SUIT-10 is 344 probes
num_probes = len(probe_expr)
print('\n********** ')
print('For atlas: ' +atlas_str+ ', number of probes = ' + str(num_probes))
print('********** \n')

probe_expr.to_csv(save_dir + 'probe_expr_cortex_'+atlas_str+'.csv')  
probe_coords.to_csv(save_dir + 'probe_coords_cortex_'+atlas_str+'.csv')  



#%% Create probe expr array w/ out corrected MNI coords 

atlas_str = 'MDTB-10'
atlas = atlas_dir+atlas_str+'.nii'
atlas_img = image.load_img(atlas)
atlas_data = atlas_img.get_fdata()
mask_data = np.zeros((atlas_data.shape))
mask_ones_cerebellum = np.where(atlas_data > 0)

mask_data[mask_ones_cerebellum] = 1

mask = image.new_img_like(atlas, mask_data, copy_header=True)

probe_expr, probe_coords = abagen.get_samples_in_mask(mask=mask,
                           data_dir = ahba_dir,
                           donors='all',
                           ibf_threshold=Parameters.ibf_threshold,
                           probe_selection=Parameters.probe_selection,
                           #donor_probes='independent',
                           sim_threshold=None,
                           lr_mirror=Parameters.lr_mirror,
                           exact=True, missing=None,
                           tolerance=3,
                           sample_norm=Parameters.sample_norm,
                           gene_norm=Parameters.gene_norm,
                           norm_matched=True,
                           norm_structures=False,  #'cortex' or 'cerebellum' #Upate: after reviews, changing this to False
                           region_agg=Parameters.region_agg,
                           agg_metric=Parameters.agg_metric,
                           corrected_mni=True,
                           reannotated=True,
                           return_counts=False,
                           return_donors=True,
                           return_report=False,
                           verbose=0) 


for key in probe_coords.keys():
    probe_coords[key]['donor'] = key 
# concatenating the DataFrames
coord_df = pd.concat(probe_coords.values())

for key in probe_expr.keys():
    probe_expr[key]['donor'] = key 
# concatenating the DataFrames
expr_df = pd.concat(probe_expr.values())

num_probes = len(coord_df)
print('\n********** ')
print('For atlas: ' +atlas_str+ ', number of probes = ' + str(num_probes))
print('********** \n')

len(np.unique(coord_df.index.values)) #unique well_id values across all donors, don't need to save donor info below

expr_df.to_csv(save_dir + 'probe_expr_cerebellum_'+atlas_str+'2.csv') 
coord_df.to_csv(save_dir + 'probe_coords_cerebellum_'+atlas_str+'2.csv')

