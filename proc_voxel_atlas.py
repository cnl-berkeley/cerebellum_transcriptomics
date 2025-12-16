#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:18:52 2023

@author: leanaking
"""
import abagen
import numpy as np
import pandas as pd
import pickle
import csv
from nilearn import image
import os

ahba_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid/AHBA/'
sub_dir = '/Users/leanaking/freesurfer/subjects/AHBA/'
project_dir = '/Users/leanaking/Projects/cerebellum_transcriptomics/'

atlas_dir = project_dir + '/external/atlas_templates/'
fig_dir = project_dir + 'leana_folder/figures/'

#%% Get cerebellum template 

# Load cerebellum  atlas as template to get cerebelum voxels 
atlas_image = atlas_dir + 'MDTB-10.nii'
atlas_csv = atlas_dir + 'MDTB-10-info.csv'
atlas_info_df = pd.read_csv(atlas_csv, header=0)
atlas_info_df = atlas_info_df.set_index('region_num')

# Load nifti
ni = image.load_img((atlas_image))
ni_data = ni.get_fdata()

# Check number of non-zero values 
nonzero = ni_data > 0 
sum_nonzero = sum(sum(sum(nonzero)))

# always check volume in freeview 
# Check if ROI values are valid
roi_ids = np.unique(ni_data)

# For MDTB atlas, rois are numbers 1-10
# size of non-zero elements = 174013
# make new list of 'voxel rois' with integers 1-174013


#%% Make voxel-wise parcellation 

# New 'voxel' roi list 
new_roi_nums = np.arange(1, sum_nonzero+1)

# initiate new empty 3d array 
temp_vol = np.zeros((256, 256, 256))

# get list of indices for each non-zero element 
inds = np.where(ni_data > 0)
x_inds = inds[0]
y_inds = inds[1]
z_inds = inds[2]

# loop through each one and replace with a new roi-label 
for i in range(sum_nonzero):
    x = x_inds[i]
    y = y_inds[i]
    z = z_inds[i]
    new_val = new_roi_nums[i]
    temp_vol[x, y, z] = int(new_val)
    
    
# Check new volume 
print( sum(sum(sum(temp_vol > 0))) )
new_roi_ids = np.unique(temp_vol)


#%% Save new cerebellum voxel-wise template 

# create new combined L&R image file 
new_img = image.new_img_like(ni, temp_vol, copy_header=True)

# Save Image
new_img.to_filename(project_dir+'leana_folder/data/cerebellum_voxel_parc.nii')


# save matching .csv atlas info file for new template - this might not be needed
# id: an integer ID corresponding to the labels in the atlas image - this is the index of the df 
# hemisphere: a left/right/bilateral hemispheric designation (i.e., ‘L’, ‘R’, or ‘B’)
# structure: a broad structural class designation (i.e., one of ‘cortex’, ‘subcortex/brainstem’, ‘cerebellum’, ‘white matter’, or ‘other’)



