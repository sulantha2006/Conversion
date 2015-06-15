__author__ = 'Sulantha'

figOutputPath = '/home/sulantha/Desktop/Classification/ROCs/'

all_list = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                'PTAU_AB142_Ratio', 'Total_TAU', 'TTAU_AB142_Ratio', 'PTAU_TTAU_Ratio', 'AV45_bl_Global_SUVR_NEW', 'FDG_bl_Global_SUVR_NEW', 'AV45_region1', 'AV45_region2',
                'AV45_region3', 'AV45_region4', 'FDG_region1', 'FDG_region2', 'FDG_region3',
                'FDG_region4', 'FDG_region5']
xticks_dict = dict(Age_bl='Age',
                   PTGENDER='Gender',
                   APOE_bin='APOE',
                   PTAU181P_bl='CSF PTAU',
                   PTAU_Pos='CSF PTAU > 23',
                   ABETA142='CSF ABETA',
                   ABETA142_Pos='CSF ABETA > 192',
                   PTAU_AB142_Ratio='PTAU ABETA Ratio',
                   Total_TAU='Total TAU',
                   TTAU_AB142_Ratio='TTAU ABETA Ratio',
                   PTAU_TTAU_Ratio='PTAU TTAU Ratio',
                   AV45_bl_Global_SUVR_NEW='AV45 Global SUVR',
                   FDG_bl_Global_SUVR_NEW='FDG Global SUVR',
                   AV45_region1='AV45 Sup. Tem. Gyrus',
                   AV45_region2='AV45 Basal Tem. Lobe',
                   AV45_region3='AV45 OFC (Basal)',
                   AV45_region4='AV45 Angular Gyrus',
                   FDG_region1='FDG Precuneus',
                   FDG_region2='FDG PCC',
                   FDG_region3='FDG OFC',
                   FDG_region4='FDG Piriform C.',
                   FDG_region5='FDG Hippocampus')
