#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:15:05 2024

Examining different gene sets from different cerebellar parcelations 

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
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster import hierarchy
from nilearn import plotting
import datetime

#%%
# Using maedbh's package
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
save_dir = project_dir + 'leana_folder/data/'


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

bad_brain_genes = ['MAGEA12', 'SPRR2B', 'MMP24', 'TNFSF18', 'FABP9', 'PGC', 'PRKACG', 'FRMPD2', 'FAM71E2', 'POU4F2', 'ALX4', 'EXD1', 'LINC00207', 'KCNK4', 'MC5R', 'PCDHGB1', 'SSX3', 'GSX2', 'MAGEC3', 'GALNT8', 'CRHR1', 'RP11-195F19.5', 'KCNA6', 'WFDC6', 'SLC35F4', 'GUCY2F', 'GSG1', 'CCDC105', 'DEFB134', 'PRY', 'BMP15', datetime.datetime(2018, 9, 8, 0, 0), 'MYH2', datetime.datetime(2018, 9, 3, 0, 0), 'NIPAL4', 'ATP1B4', 'GLB1L3', 'PLSCR5', 'ARMC3', 'GJB5', 'HTR1D', 'NKX2-1', 'GAL3ST2', 'GIF', 'IL25', 'OR13C3', 'LRTM1', 'TRHR', 'LRRC52', 'PAX1', 'DEFA5', 'AGXT2', 'TSGA13', 'ACTL7A', 'PNPLA5', 'C10orf126', 'ARID3C', 'PIWIL1', 'KRT77', 'PIWIL3', 'ZSWIM2', 'DHRS7C', 'BRS3', 'FEZF1', 'MS4A12', 'RXFP3', 'FAM170A', 'ACPT', 'MAGEC1', 'C12orf40', 'VSX2', 'TMIGD1', 'OR4N5', 'ZNF804B', 'SI', 'TEX19', 'TRPC7', 'SPATS1', 'IFNW1', 'NPAS4', 'TP53TG3', 'SP8', 'NPBWR2', 'FBXW12', 'MRGPRX1', 'SLC34A1', 'HELT', 'NEUROD4', 'RTL1', 'GPR151', 'DMRTC1B', 'PAX2', 'TRIM43', 'ATP13A5', 'SMCP', 'ELSPBP1', 'HSF2BP', 'PTF1A', 'OR1G1', 'PGA3', 'SPATA22', 'THEG', 'LCN1', 'WNT8A', 'LECT1', 'ZNF560', 'GUCY1B3', 'C3orf49', 'TRIM48', 'C22orf24', 'SLC22A8', 'TBX15', 'PATE1', 'TNP1', 'LPA', 'CHRNB3', datetime.datetime(2018, 9, 5, 0, 0), 'CLRN1', 'FAM47A', 'CSMD2', 'ATP4B', 'ZIC2', 'PAX7', 'OCM2', 'RGAG1', 'C1orf64', 'GJD4', 'ADAMTS20', 'OR2K2', 'CRYGB', 'TAAR6', 'KNCN', 'PITPNM3', 'VCX3A', 'PBOV1', 'C4orf22', 'PAK6', 'GKN2', 'XKR3', 'TMC3', 'SLC7A13', 'CATSPER4', 'HTR1B', 'DAND5', 'C17orf105', 'LPO', 'OR2T8', 'SLC13A1', 'ANKRD62', 'C12orf50', 'SPACA5', 'PCDHAC1', 'C22orf42', 'C1orf87', 'EVX2', 'TSPYL6', 'OR8D4', 'YIPF7', 'PCDHGA5', 'OR2M4', 'AMBN', 'GDF1', 'CRYBA1', 'TRPM8', 'ITIH2', 'TMEM89', 'SALL3', 'LHX5', 'TMPRSS11F', 'ATOH1', 'ANKRD34B', 'FERD3L', datetime.datetime(2018, 3, 1, 0, 0), 'PHKG1', 'VSTM5', 'SAGE1', 'BTG4', 'ACOT12', 'GNAT1', 'C10orf120', 'MC4R', 'THEM5', 'FBXO47', 'OR13J1', 'GRK1', 'GPR50', 'CRYAA', 'FATE1', 'OR6A2', 'DNAH17', 'OR4F21', 'DDX4', 'HGFAC', 'C2orf83', 'CCKAR', 'DNMT3L', 'GDF5', 'ADAMTS16', 'DMBX1', 'ALPI', 'DRD3', 'SERPINB13', 'GUCA1A', 'TECTB', 'NPFFR1', 'PCDHA3', 'VN1R2', 'CACNG5', 'GPR21', 'DMRT1', 'FGF22', 'MAGEA4', 'RS1', 'PCDHB6', 'MKRN3', 'MAS1', 'RHOXF2B', 'WNT1', 'CBS', 'MC3R', 'IRS4', 'UROC1', 'NLRP10', 'FAM153B', 'MSLNL', 'C2orf82', 'TSPAN19', 'KIAA2022', 'OR1L3', 'HORMAD2', 'GNAT3', 'PALM2-AKAP2', 'LECT2', 'KCNG4', 'IGFL1', 'TTC29', 'GJD2', 'DMRT3', 'FSHB', 'HMSD', 'OR1F1', 'ARR3', 'GUCA1C', datetime.datetime(2018, 9, 4, 0, 0), datetime.datetime(2018, 9, 12, 0, 0), 'ZAR1L', 'OR4N2', 'KPRP', 'SEC14L3', 'OR1Q1', 'RNASE2', 'OR5B12', 'CDCP2', 'ACRV1', 'PRAMEF2', 'PCDHGB6', 'DBNDD2', 'IQCF3', 'ROPN1B', 'SLC6A11', 'CHI3L2', 'OR2L2', 'GABRA6', 'OPTC', 'OTOL1', 'IFNA6', 'SLC36A2', 'IL1A', 'OR2T33', 'TRIML2', 'WNT9B', 'PNMAL2', 'FRMPD1', 'C10orf62', 'LCE1E', 'OR7D2', 'RXFP2', 'SGCZ', 'BEND4', 'PRAMEF1', 'FGF5', 'FGF11', 'CHRNA9', 'PRB2', 'HAO1', 'C7orf34', 'OPN5', 'ZFP57', 'EFCAB5', 'SAG', 'RFPL4B', 'ART1', 'GPR45', 'PTH2R', 'TRPV5', 'GLP1R', 'HSFY1', 'TRIM54', 'KIAA1024L', 'SLC26A9', 'DUSP27', 'NHLH2', 'PCDHA9', 'OR7A5', 'AVPR1B', 'GPRASP2', 'FOXD4L3', 'NDST4', 'TEX15', 'FAM9C', 'OR9A2', 'OR10G4', 'NEUROG2', 'GLRA1', 'MYH4', 'PSG9', 'C14orf37', 'C1QL4', 'STARD6', 'RP11-770J1.4', 'C4orf50', 'LRIT2', 'OR51B4', 'INSL5', 'GRM6', 'HCRTR2', 'WDR87', 'AMTN', 'IBSP', 'PRAMEF8', 'ZNF804A', 'SLC6A5', 'IFNA1', 'C1orf185', 'TINAG', 'PCDHB1', 'TMC2', 'OR1L1', 'LRIT1', 'GCM2', 'OR10H1', 'SLC6A18', 'TMEFF1', 'AKAP14', 'OR52I1', 'FOXB2', 'NHLH1', 'CCDC33', 'C9orf135', 'PCDHGA4', 'GBX1', 'CAPSL', 'MMD2', 'AIPL1', 'NOG', 'STK32A', 'DRGX', 'UBL4B', 'PROK1', 'TTLL9', 'SLC14A2', 'MUC22', 'GHRH', 'LGALS16', 'PRG3', 'DSCR8', 'TPH2', 'C2orf27B', 'MAGEA1', 'RP11-566K11.2', 'GBX2', 'C10orf53', 'FETUB', 'TAS2R1', 'KRT33B', 'PCDHA8', 'ROS1', 'MSTN', 'HFE2', 'DEFB112', 'TP53TG3B', 'GALNTL6', 'SIGLEC8', 'MAGED4B', 'XAGE1B', 'CSN1S1', 'ROPN1', 'C5orf47', 'SPRR2D', 'ERAS', 'PRAMEF12', 'AQP12A', 'SPAG6', 'PRODH2', 'KCNG2', 'LRRC9', 'PLA2G2C', 'KRT75', 'IGFL4', 'GRHL3', 'C7orf65', 'MEPE', 'LRRC14B', 'SMYD1', 'C1orf141', 'LIPF', 'ANKFN1', 'SLC38A8', 'C10orf35', 'NT5C1A', 'DAO', 'ADAMTS18', 'OPRD1', 'KIR3DX1', 'OR2T4', 'OR5D18', 'LPAR4', 'PSG11', 'GAGE12G', 'PNMAL1', 'LYG2', 'FCGR1A', 'CCL1', 'MYL1', 'SLC22A25', 'PAH', 'C17orf74', 'BHMT', datetime.datetime(2018, 3, 11, 0, 0), 'GLT6D1', 'ADIG', 'DRD5', 'ARL5C', 'TEKT1', 'CGB2', 'CACNA1S', 'KIAA1644', 'MCCD1', 'MURC', 'OR4N4', 'ZCCHC16', 'OR52K1', 'SPRR2G', 'FOXE1', 'OBP2B', 'COL20A1', 'OR10A4', 'PCDHGA1', 'FAM71C', 'C8orf86', 'C17orf77', 'IL9', 'FNDC8', 'KCNK18', 'GPR101', 'TCEB3B', 'GLRA4', 'TAS2R3', 'CD200R1L', 'WBSCR17', 'NPSR1', 'RPE65', 'SSX1', 'OR51G2', 'UGT1A9', 'AGBL1', 'MORC1', 'PSG8', 'MTNR1B', 'KIAA1456', 'SLC7A3', 'C21orf62', 'SPINK9', 'OR5AK2', 'VAX1', datetime.datetime(2018, 3, 4, 0, 0), 'CECR6', 'LCN9', 'HTR6', 'NEU2', 'ACOT6', 'LYZL4', 'PCDHA13', 'SRY', 'MUC19', 'KLRC4', 'CNGA3', 'GAGE10', 'C14orf178', 'SLC12A1', 'HIPK4', 'RHOXF2', 'CCDC27', 'SEMG1', 'TMEM8C', 'SNRPN', 'FNDC7', 'PROKR2', 'DLX6', 'TACR3', 'PCDHGA9', 'OR2T10', 'SPATA19', 'SLC22A6', 'STRA8', 'RGS21', 'TSPAN16', 'KCNU1', 'OR13D1', 'ODF1', 'LHFPL1', 'DCLK3', 'PRAMEF7', 'KCNA7']
filt_brain_genes = list(set(brain_genes) - set(bad_brain_genes))



#%% Get Purkinje Cell Marerks

# Aldinger et al. Nat Neuro (2021)
purkinje_markers = ['CALB1', 'CA8', 'ITPR1', 'FOXP2', 'BCL11A', 'SKOR2'] # from supp table 
fig2E_genes = ['RSPO1','WNT2B','RSPO3','SLFN13', 'CALCB','ATP6V1C2','CALCA',
                    'DPYD','NEUROG1','OR4M2','OR4N3P','ATOH1','MGP','UCN3','DYTN',
                    'NKX3–2','DAPL1','GGNBP1','MCOLN3', 'ITPKA','IL22','IGF1',
                    'NRK','SMPX', 'CALB1', 'PRSS22','SST','TLL2','KRT74']

#%% Get Cyo genes
 
cyto_dir = '/Users/leanaking/Projects/julich_transcriptomics_valid'
version = 'julich/v29'
cytodata_dir = (cyto_dir + '/data/'+version)
title_str = 'PC1 ALL Brain-specific'  
cond_str = 'all' #'PC1_200_BS'

with open((cytodata_dir+'/PC1_output_clustered_ABGVOL_all_BS.pickle'), 'rb') as handle:
    data = pickle.load(handle)
#top_cortical_genes = data['top_gene_expr'] 
cyto_genes = data['top_gene_name']


with open((cytodata_dir+'/dge_output_clustered_ABGVOL_top200_allbrain.pickle'), 'rb') as handle:
    data = pickle.load(handle)
#top_cortical_genes = data['top_gene_expr'] 
cyto_genes = data['top_gene_name']




#%% 1. Get top 1% gene set for each parcellation 
# and BS-filtered genes 

parcellations = ['SUIT-10', 'Buckner-7', 'Buckner-17', 'MDTB-10', 'Yeo-7', 
                 'Yeo-17', 'MDTB-10-subRegions', 'Ji-10', 'Desikan-Killiany-83', 'SUIT-26'] 

parcellations_bs = ['SUIT-10', 'Buckner-7', 'Buckner-17', 'MDTB-10', 'Yeo-7', 'MDTB-10-subRegions'] 

def get_gene_set(percentile, which_genes, atlas):
    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, 
                                 remove_outliers=True, normalize=True)
    df = df.set_index(df.index)
    gene_set = df.index.values.tolist()
    return df, gene_set 

top1_dfs = {key: None for key in parcellations}
top1BS_dfs = {key: None for key in parcellations_bs}

top1_sets = {key: None for key in parcellations}
top1BS_sets = {key: None for key in parcellations_bs}


percentile = 1
gene_filt = ['top_bs', 'top'] 
for which_genes in gene_filt:
    if which_genes == 'top':
        for atlas in top1_dfs:
            df, gene_set = get_gene_set(percentile, which_genes, atlas)
            top1_dfs[atlas] = df
            top1_sets[atlas] = gene_set
    elif which_genes == 'top_bs':
        for atlas in top1BS_dfs:
            df, gene_set = get_gene_set(percentile, which_genes, atlas)
            top1BS_dfs[atlas] = df
            top1BS_sets[atlas] = gene_set
    else:
        print('Error')
        


#%% 2. Compare get lists 
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted

def get_BurtCat_overlap(cat_genes, gene_name):
    brain_inds = []
    for name in gene_name:
        if name in cat_genes: ### filtering to brain-RELATED genes
            ind = gene_name.index(name)
            #print(ind)
            brain_inds.append(ind)
        else:
            pass
    bs_gene_name = [gene_name[i] for i in brain_inds]
    gene_len = len(gene_name)
    percent = len(bs_gene_name)/gene_len * 100 
    
    return bs_gene_name, percent


cat_genes = brain_genes 
for atlas, gene_set in top1_sets.items():
    bs_gene_name, percent = get_BurtCat_overlap(cat_genes, gene_set)
    print(str(atlas) + ' overlap with brain-specific genes = '+ str(round(percent, 2)) + '%')



# Get overlap for pairs 
def get_vennplot(set1, set2, set_labels, cond): 
    plt.figure(figsize=(4,4))
    vd2=venn2([set(set1), set(set2)],
              set_labels = set_labels,
              set_colors=('lightblue', 'mediumblue'),#('mediumblue', 'crimson')
              alpha = 0.45)
    venn2_circles([set(set1), set(set2)],
                  linestyle='-', linewidth=0.5, color='black')
    #plt.title('Differential Brain Gene Overlap', fontweight='bold', fontsize=18,
    #          pad=1, color='black');
    #for t in vd2.set_labels: t.set_fontsize(16)
    #for t in vd2.subset_labels: t.set_fontsize(16)
    #for t in vd2.subset_labels: t.set_fontweight('bold')
    if cond == 'top':
        plt.savefig(fig_dir+'/gene_sets/venn_diagrams/'+set_labels[0]+'_'+set_labels[1], dpi=300, 
                   bbox_inches='tight', transparent=True)
    elif cond == 'top_bs':
        plt.savefig(fig_dir+'/gene_sets/venn_diagrams_bs/'+set_labels[0]+'_'+set_labels[1], dpi=300, 
                   bbox_inches='tight', transparent=True)
    plt.show()
    
    
cond = 'top'
for atlas1, set1 in top1_sets.items():
    for atlas2, set2 in top1_sets.items():
        if atlas1 == atlas2:
            continue
        else:
            set_labels = [atlas1, atlas2]
            get_vennplot(set1, set2, set_labels, cond)
        
        
cond = 'top_bs'
for atlas1, set1 in top1BS_sets.items():
    for atlas2, set2 in top1BS_sets.items():
        if atlas1 == atlas2:
            continue
        else:
            set_labels = [atlas1, atlas2]
            get_vennplot(set1, set2, set_labels, cond)
            
            
# plot MDTB-10 norm vs. filt venn diagram 
cond = 'normVfilt'  
atlas1 = 'MDTB-10-subRegions'
set1 = top1_sets[atlas1]   
set2 = top1BS_sets[atlas1]  
set_labels = ['No Filtering', 'Cortex-Specific\n Filtering']

plt.figure(figsize=(4.3,4.3))
font1 = {'family':'Helvetica','size':16} # use for title
plt.rc('font', **font1)
vd2=venn2([set(set1), set(set2)],
          set_labels = set_labels,
          set_colors=('lightblue', 'mediumblue'),#('mediumblue', 'crimson')
          alpha = 0.50)
venn2_circles([set(set1), set(set2)],
              linestyle='-', linewidth=1, color='black')
#plt.title((atlas1+' Top Gene Set Overlap'), fontweight='bold', fontsize=14,
#          pad=1, color='black');
plt.savefig(fig_dir+'/gene_sets/venn_diagrams/normVfilt_'+atlas1+'.png', dpi=500, 
           bbox_inches='tight', transparent=True)



# Get venn diagram as 'pie char as bar
# total_gene = len(top_cyto_genes)
# cyto_glssr_overlap = set.intersection(set(top_cyto_genes), set(top_glasser_genes))
# cyto_only = list(set(top_cyto_genes) - set(top_glasser_genes)) # unique elements of list 2
# glssr_only = list(set(top_glasser_genes) - set(top_cyto_genes)) 


# category_names = ['Cytoarchitectonic', 'Overlap', 'Multimodal']
# results = {
#     'Top 10%': [len(cyto_only), len(cyto_glssr_overlap), len(glssr_only)]
# }


# labels = list(results.keys())
# data = np.array(list(results.values()))
# data_cum = data.cumsum(axis=1)
# category_colors = plt.get_cmap('Blues')(np.linspace(0.15, 0.85, data.shape[1]))
# #category_colors = ['lightblue', '', 'mediumblue']


# fig, ax = plt.subplots(figsize=(8, 2))
# ax.invert_yaxis()
# ax.xaxis.set_visible(False)
# ax.set_xlim(0, np.sum(data, axis=1).max())
# #
# for i, (colname, color) in enumerate(zip(category_names, category_colors)):
#     widths = data[:, i]
#     starts = data_cum[:, i] - widths
#     ax.barh(labels, widths, left=starts, height=0.5,
#             label=colname, color=color)
#     xcenters = starts + widths / 2
#     r, g, b, _ = color
#     text_color = 'white' if r * g * b < 0.5 else 'black'
#     for y, (x, c) in enumerate(zip(xcenters, widths)):
#         ax.text(x, y, str(int(c)), ha='center', va='center',
#                 color=text_color, fontsize=12, fontweight='semibold')
# ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
#           loc='lower left', fontsize='small')



#%% 3. Get similarity matrices 

cortical_parcellations = ['Yeo-7', 'Yeo-17', 'Desikan-Killiany-83']
cerebellar_parcellations = ['SUIT-10', 'SUIT-26', 'MDTB-10', 'MDTB-10-subRegions', 
                            'Buckner-7', 'Buckner-17', 'Ji-10']
both = ['Yeo-Buckner-7', 'Yeo-Buckner-17']


percentile = 1
which_genes = 'top' #top or top_bs
#atlas_other = "MDTB-10"
# make matrix for each atlas
for parc in cerebellar_parcellations:
    df = ana.return_grouped_data(atlas=parc, which_genes=which_genes, percentile=percentile, 
                                 reorder_labels=False, remove_outliers=True, normalize=True)
    visualize.simple_corr_heatmap(df, atlas=parc, distance_correct=True)#, ax=ax1)
    plt.savefig(fig_dir+'gene_sets/corr_matrices/'+parc+'_'+which_genes, dpi=300, 
               bbox_inches='tight', transparent=True)
    plt.clf()



# Maedbh's original plot: 
which_genes = 'top' #top or top_bs
atlas_other = "MDTB-10" 
atlas = 'MDTB-10-subRegions'
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, reorder_labels=True, remove_outliers=True, normalize=True)
visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True)#, ax=ax1)
plt.savefig(fig_dir+'gene_sets/corr_matrices/'+atlas_other+'_on_'+atlas+'_'+which_genes, dpi=300, 
               bbox_inches='tight', transparent=True)

# My original plot
which_genes = 'top_bs' #top or top_bs
atlas_other = "MDTB-10" 
atlas = 'MDTB-10-subRegions'
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, reorder_labels=True, remove_outliers=True, normalize=True)
visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True)#, ax=ax1)
plt.savefig(fig_dir+'gene_sets/corr_matrices/'+atlas_other+'_on_'+atlas+'_'+which_genes, dpi=300, 
               bbox_inches='tight', transparent=True)


# Comparing suit diff genes on MTBD parc 
which_genes = 'top' #top or top_bs
atlas = 'MDTB-10-subRegions'
atlas_other = "SUIT-10" 
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, reorder_labels=True, remove_outliers=True, normalize=True)
visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True)#, ax=ax1)
plt.savefig(fig_dir+'gene_sets/corr_matrices/'+atlas_other+'_on_'+atlas+'_'+which_genes, dpi=300, 
               bbox_inches='tight', transparent=True)


# Comparing suit diff genes on MTBD parc 
which_genes = 'top' #top or top_bs
atlas = "SUIT-10" 
atlas_other = "MDTB-10-subRegions" 
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, reorder_labels=False, remove_outliers=True, normalize=True)
visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True)#, ax=ax1)
plt.savefig(fig_dir+'gene_sets/corr_matrices/'+atlas_other+'_on_'+atlas+'_'+which_genes, dpi=300, 
               bbox_inches='tight', transparent=True)
plt.clf()


# make matrix for each atlas compared to the cortical parcellation 
plt.clf()
atlas_other="SUIT-10"
which_genes = 'top' #top or top_bs
df = ana.return_concatenated_data(atlas_cerebellum="Buckner-17", atlas_cortex="Yeo-17", which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=True, normalize=True)
visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-17", distance_correct=True, simple_labels=True)# ax = ax2)
plt.savefig(fig_dir+'gene_sets/corr_matrices/cortcereb_'+atlas_other+'_'+which_genes, dpi=300, 
               bbox_inches='tight', transparent=True)
    


#%% 4. Get PCs

# Get variance explained by first PC
percentile = 1
atlas='SUIT-10'
atlas_other= 'SUIT-10' #"MDTB-10"
which_genes='top'
remove_outliers=True
normalize=True
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, 
                             atlas_other=atlas_other, remove_outliers=remove_outliers, 
                             normalize=normalize)
ana._variance_explained(df, pcs=[1])


# 3e
group_pcs=False
pcs=[1]
cmap, labels = visualize._make_colormap(atlas)
#cpal = visualize._make_colorpalette("mycolormap")
#mpl.cm.register_cmap("mycolormap", cmap)
cpal = sns.color_palette("mycolormap", n_colors=len(labels))
# gets the correct color for each region (only a problem when there are missing regions)
#labels, cpal_reordered = visualize._reorder_colors_x_axis(df, cpal)
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
plt.savefig(fig_dir+'gene_sets/PC_plots/'+atlas_other+'_on_'+atlas+'_'+which_genes, dpi=300, 
               bbox_inches='tight', transparent=True)
plt.tight_layout()




### Circular Plot ###
pc_df = pd.DataFrame(data=vt[pcs[0], :], index=labels, columns=['PC1'])
# set figure size
plt.figure(figsize=(20,10))

# plot polar axis
ax = plt.subplot(111, polar=True)

# remove grid
plt.axis('off')

# Set the coordinates limits
upperLimit = 100
lowerLimit = 30

# Compute max and min in the dataset
max = pc_df['PC1'].max()

# Let's compute heights: they are a conversion of each item value in those new coordinates
# In our example, 0 in the dataset will be converted to the lowerLimit (10)
# The maximum will be converted to the upperLimit (100)
slope = (max - lowerLimit) / max
heights = slope * pc_df.PC1 + lowerLimit

# Compute the width of each bar. In total we have 2*Pi = 360°
width = 2*np.pi / len(pc_df.index)

# Compute the angle each bar is centered on:
indexes = list(range(1, len(pc_df.index)+1))
angles = [element * width for element in indexes]
angles

# Draw bars
bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=2, 
    edgecolor="white")




#%% 5. Get 3-D plots

# 1. Choose atlas mask (for probes selected)
atlas_mask = 'MDTB-10' # parcellations or 'andcortex' 
structure = 'cerebellum' #cortex or cerebellum



# 2. Choose gene set 
# if from parcellation:
other = False 
geneset_str = 'MDTB-10' #'MDTB-10-subRegions'
filt = 'top' # 'topbs' or 'top' (no filter)

# if from previously defined list of genes:
#other = True 
#other_set = cyto_genes #filt_brain_genes
#geneset_str = 'Cyto_BS' #'MDTB-10-subRegions'
#filt = '' # 'topbs' or 'top' (no filter)

# 1. use MNI coords from 'mni_coords_all_donors.csv'
probe_expr = pd.read_csv(save_dir + 'probe_expr_'+structure+'_'+atlas_mask+'2.csv', index_col=0)  
probe_coords = pd.read_csv(save_dir + 'probe_coords_'+structure+'_'+atlas_mask+'2.csv', index_col=0)  
well_ids = probe_expr.index.to_list()

# Update post-review - try to match well_ids first then get flatmap plot? 
#both_coords = probe_coords.merge(raw_coords, how='left', on='well_id')



# 2. then sub-select for genes 
if other == False:
    if filt == 'topbs':
        gene_set = top1BS_sets
    else:
        gene_set = top1_sets
        
    gene_set_using = gene_set[geneset_str]
    if geneset_str == 'MDTB-10-subRegions':
        if filt == 'topbs':
            gene_set_using.remove('C4orf22')
    else:
        pass
    set_expr = probe_expr[gene_set_using].copy()
elif other == True:
    # if using non-parc set:
    set_expr = probe_expr[other_set].copy()

# or just load in the list here: 
atlas = 'MDTB-10'
which_genes = 'top'
percentile = 1
df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, 
                             remove_outliers=True, normalize=True)
df = df.set_index(df.index)
gene_set = df.index.values.tolist()
set_expr = probe_expr[gene_set].copy()

### Clustering for probe data ### 
# a. Get similarity matrix 
similarities = euclidean_distances(set_expr)
#plot
fig = plt.figure(figsize=(15,10))
ax = plt.gca()
ax = sns.heatmap(similarities, xticklabels=well_ids, yticklabels=well_ids,
                 cbar_kws={'label': 'Euclidean Dissimarlity Distance'},
                 cmap="OrRd")
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12) 
ax.figure.axes[-1].yaxis.label.set_size(16)
ax.tick_params(axis='both', labelsize=0) #12
plt.show()
fig.savefig((fig_dir+'gene_sets/probe_clustering/dissimarlity_mat_'+atlas_mask+'_'+structure+'_genelistfrom_'+geneset_str+'_'+filt),
            dpi=300, bbox_inches='tight', transparent=False)

# b. Hierarchical Clustering (to determine num clusters for MDS)
#
color_threshold = 80
#
Z = hierarchy.linkage(set_expr, method='ward', metric='euclidean')
fig = plt.figure(figsize=(6, 5))
dn = hierarchy.dendrogram(Z, leaf_rotation=60, leaf_font_size=0, labels=well_ids, 
                          distance_sort=True, color_threshold=color_threshold, above_threshold_color='k')
plt.xlabel("Probes", fontsize=14, fontweight='bold')
plt.ylabel("Euclidean Distance", fontsize=14, fontweight='bold')
#plt.grid(which = 'major', color='k', linewidth=0.5, alpha=0.2)
plt.show()
fig.savefig((fig_dir+'gene_sets/probe_clustering/dendo_'+atlas_mask+'_'+structure+'_genelistfrom_'+geneset_str+'_'+filt),
            dpi=300, bbox_inches='tight', transparent=False)



plt.figure(num=1, figsize=[7,7])
set_link_color_palette(['orange', 'green', 'red'])
with plt.rc_context({'lines.linewidth': 1.5}):
    R = dendrogram(
            Z=linkage(set_expr, method='ward', metric='euclidean'),
            orientation='top',
            get_leaves=True,
            color_threshold=80,
            labels=set_expr.index.to_list(),
            distance_sort='ascending',
            above_threshold_color='black')
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.tick_params(axis='y', which='major', labelsize=25)
plt.xlabel('Microarray Samples')
plt.ylabel("Euclidean Distance")
plt.savefig((fig_dir+'gene_sets/probe_clustering/dissimarlity_mat_'+atlas_mask+'_'+structure+'_genelistfrom_'+geneset_str+'_'+filt+'.png'), 
            dpi=600, bbox_inches='tight', transparent=False)

### lets try kmeans for the heck of it since this is probe clustering
## maybe hierarchical clustering isn't the best approach for probe data?
from sklearn.cluster import KMeans
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(set_expr)
    set_expr["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()



###
num_clusters = 3
###
# c. MDS Then get clustering labels for probes 
# Step 1 (using WARD to get clustering and MDS on ditance matrix)
model = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward') 
model.fit_predict(set_expr)
clusters = model.labels_.tolist()
# Step 2 - using MDS to get 2d ROI distances
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1) #n_components=2
pos = mds.fit_transform(similarities)  # shape (n_components, n_samples)
xs, ys = pos[:, 0]*-1, pos[:, 1]*-1
#
#cluster_colors = {0: 'green', 1:'red'} # 2 cluster option
cluster_colors = {0: 'orange', 1:'red', 2:'green'} # 3 cluster option
#cluster_colors = {0: 'purple', 1:'green', 2:'orange', 3: 'red'} # 4 cluster option
#cluster_colors = {0: 'blue', 1:'red', 2:'orange', 3: 'purple', 4: 'green'} # 5 cluster option
#cluster_colors = {0: 'orange', 1:'darkorange', 2:'darkolivegreen', 3: 'lightgreen', 4:'green'} # 5 cluster cyto option
#cluster_colors = {0: 'orange', 1:'green', 2:'red', 3: 'purple', 4:'blue', 5:'cyan'} # 6 cluster  option
#
cluster_names = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5:'Cluster 6'}
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=well_ids)) 
#group by cluster
groups = df.groupby('label')
df.astype(bool).sum(axis=0)
# MDS Plot
fig, ax = plt.subplots(figsize=(8, 7)) # set size
#iterate through groups to layer the plot
# Plotting color coded points
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='none', ms=5, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none') #ms=5
    ax.set_aspect('auto')
ax.grid(which = 'major', color='k', linewidth=0.5, alpha=1, linestyle='-')
ax.set_title('MDS', fontsize=16, fontweight='bold')    
plt.xticks(fontsize=13)    
plt.yticks(fontsize=13)    
ax.set_xlabel('Euclidean Distance', fontsize=14)
ax.set_ylabel('Euclidean Distance', fontsize=14)
ax.grid(True)
ax.grid(which = 'major', color='k', linewidth=0.5, alpha=0.5, linestyle='--')
ax.grid(which = 'minor', color='k', linewidth=0.5, alpha=0.5, linestyle='--')
fig.savefig((fig_dir+'gene_sets/probe_clustering/MDS_'+atlas_mask+'_'+structure+'_genelistfrom_'+geneset_str+'_'+filt),
            dpi=300, bbox_inches='tight', transparent=False)
plt.show() 

## TODO: d. try K-means clustering to look at difference (start w/ topbs on MDTB-10 gene set)




### 3-D plotting (HTML) ###
probe_coords['mds_labels'] = clusters
probe_groups = probe_coords.groupby('mds_labels')
probe_coords['mds_color'] = probe_coords['mds_labels'].apply(lambda x: cluster_colors[x])

mni_coords = probe_coords[["x", "y", "z"]].to_numpy()
mni_markers = probe_coords['mds_color'].to_list()

# Plotting 
view = plotting.view_markers(mni_coords, mni_markers,
                             marker_size=7)
view.open_in_browser()
#view.save_as_html(fig_dir+'gene_sets/probe_clustering/3Dmarkers_'+atlas_mask+'_'+structure+'_genelistfrom_'+geneset_str+'_'+filt+'.html')
view.save_as_html(fig_dir+'flatmap_top1_3clust_updatednorm.html')


#%% 6. Get flatmap plots 
# makes csv file to be imported in matlab 
import SUITPy.flatmap as flatmap
import nitools as nt
save_dir = project_dir + '/leana_folder/data/'


# 1. make a version 'mni_coords_all_donors', but instead of donor id, use cluster ids
# in RGB value
probe_coords['mds_labels'] = probe_coords['mds_labels']+1
probe_coords['R'] = probe_coords['mds_color'].apply(lambda x: mpl.colors.to_rgb(x)[0])
probe_coords['G'] = probe_coords['mds_color'].apply(lambda x: mpl.colors.to_rgb(x)[1])
probe_coords['B'] = probe_coords['mds_color'].apply(lambda x: mpl.colors.to_rgb(x)[2])


# 2. update coords to match that from original csv file 
    # this doesnt have the well_id associated with it, need to get that value add to this csv 
        # gonna have to loop through sample annot for each subject 
raw_coords = pd.read_csv(project_dir + 'raw/mni_coords_all_donors.csv', index_col=0)  
raw_coords = raw_coords.reset_index(names='donor_row')
raw_coords['well_id'] = 0
donors = raw_coords['donor_id'].unique()
# start with one donor 
for donor in donors:
    file_str = project_dir + 'raw/allenbrain/normalized_microarray_' + donor + '/SampleAnnot.csv'
    annot_df = pd.read_csv(file_str)
    donor_inds = raw_coords.index[raw_coords['donor_id'] == donor].tolist()
    for i in donor_inds:
        donor_row = raw_coords.iloc[i]
        row_num = donor_row['donor_row']
        x_true = donor_row['mni_x']
        annot_row = annot_df.iloc[row_num]
        well_id = annot_row['well_id']
        x_test = annot_row['mni_x']
        if x_true == x_test:
            raw_coords.loc[i, 'well_id'] = well_id
        else:
            print('Error - coords not aligned')
            
# 3. merge df's 
both_coords = probe_coords.merge(raw_coords, how='left', on='well_id')
both_coords = probe_coords.copy()

# Delete row with nans
both_coords = both_coords.dropna() 

# save df for matlab figure 
#both_coords.to_csv(save_dir + 'probe_coords_clust_'+structure+'_'+atlas_mask+'_geneset_'+geneset_str+'_'+filt+'_matched2.csv')  
both_coords.to_csv(save_dir + 'probe_coords_clust_'+structure+'_'+atlas_mask+'_geneset_'+geneset_str+'_'+filt+'_3clust_updatednorm.csv')  

# Alignment fixed, but still missing a few probes
    # theres probably some probs missing in my data from the abagen package compared to mave's original probe list
    # missing a bunch of probes in that middle string area - most likely something to do with the mask on the abagen package
    # try re-creating the probe_coords and expr array changing these abagen parameters: 
            #Parameters.tolerance
            #Parameters.exact 

