"""
Script for creating lookup table for clustering.
"""

import sys 
import os 

import numpy as np 

import matplotlib.pyplot as plt 

import re

import pandas as pd 

# Matching names.
match_names = [('2021-06-08-14-54-59', ('20210608-f1', 'WT')), ('2021-06-25-15-21-37' ,('20210625-f2', 'HOM')), ('2021-07-06-13-48-55', ('20210706-f1', 'WT')), ('2021-07-13-12-49-43', ('20210713-f1', 'WT')),
('2021-07-16-11-32-30', ('20210716-f1', 'WT')), ('2021-08-27-11-46-32' ,('20210827-f1', 'HOM')), ('2021-09-21-11-37-35', ('20210921-f1', 'WT')), ('2021-10-13-10-44-34', ('20211013-f1', 'HOM')),
('2021-11-18-14-06-44', ('20211118-f11', 'WT')), ('2022-01-20-11-21-57' ,('20220120-f10', 'HOM')), ('2022-02-08-10-21-23', ('20220208-f10', 'WT')), ('2022-02-08-12-42-27', ('20220208-f11', 'WT')),
('2022-02-10-11-11-50', ('20220210-f10', 'HOM')), ('2022-02-16-09-46-03' ,('20220216-f10', 'WT')), ('2022-02-18-13-11-27',('20220218-f11', 'WT')), ('2022-02-18-15-06-54', ('20220218-f12', 'WT')),
('2022-03-14-09-07-13', ('20220314-f10', 'HOM')), ('2022-03-14-12-12-26' ,('20220314-f11', 'HOM')), ('2022-03-15-09-29-12',('20220315-f10', 'HOM')), ('2022-03-15-13-49-02', ('20220315-f12', 'HOM')),
('2022-03-16-14-04-55', ('20220316-f12', 'HOM')), ('2022-03-18-11-40-24' ,('20220318-f11', 'HOM')), ('2022-03-18-14-53-12',('20220318-f12', 'HOM')), ('2022-03-21-14-27-34', ('20220321-f11', 'HOM')),
('2022-03-22-14-03-22', ('20220322-f10', 'WT')), ('2022-03-22-11-59-19' ,('20220322-f11', 'WT')), ('2022-03-23-10-11-32',('20220323-f10', 'WT')), ('2022-03-23-12-10-46', ('20220323-f11', 'WT')),
('2022-03-23-14-01-19', ('20220323-f12', 'WT')), ('2022-03-24-09-04-01' ,('20220324-f10', 'WT'))]

if __name__ == '__main__':
	
	if sys.platform == 'linux' or sys.platform == 'linux2':
		main_folder = "/home/enrique/WashU/Data"
	#matplotlib.use('Agg')
	elif sys.platform == 'darwin':
		main_folder = "/Users/enriquehansen/Data/WashU/Data"
		matplotlib.use('MacOSX')
	elif sys.platform == 'win32':
		main_folder = r"C:\Users\enriq\Data\WashU\Data"
		locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
	else:
		raise NameError("Unknown OS.")
	
	columns = ['date', 'fish_num', 'clutch-num', 'gravel_reared', 'dpf', 'genotype', 'pre-processed', 'annotated', 'tracked', 'Group', 'notes']
	
	# Clear all images.
	plt.close('all')
	
	preds_folder = '/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48/video_preds/'
	
	files      = []
	for p, d, f in os.walk(preds_folder):
		for file in f:
			if file.endswith('.csv'):
				if file.endswith('_temporal_norm.csv'):
					pass
				else:
					files.append(os.path.join(p, file))

	dest_fld = os.path.join(preds_folder ,'fish-plots');
	if not os.path.exists(dest_fld):
		os.makedirs(dest_fld)
	
	bouts_folder  = preds_folder + 'bouts-data/'
	degree_folder = preds_folder + 'degree-data/'
	
	filenames = next(os.walk(bouts_folder))[2]
	
	# Load XLS file with information about all fish.
	xls_file  = [xls for xls in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, xls)) and 'zf-lsm' in xls]
	fish_info = pd.read_excel(os.path.join(main_folder, xls_file[0]))
	
	df = [] 
	for file in filenames:
		fish_name = file.replace('-bouts.mat', '')
		# There is a difference between the name of the Video file and the one of the Calcium file.
		# Because of this, we look for the Calcium file that matches the Video files.
		indx  = np.argwhere(np.array([x[0] for x in match_names]) == fish_name).item()
		f_name   = match_names[indx][1][0]
		dpf      = fish_info.loc[(fish_info['date']== int(''.join(c for c in f_name[:f_name.find('-')] if c.isdigit()))) & (fish_info['fish-num'] == int(''.join(c for c in f_name[(f_name.find('-')+2):]))), 'dpf'].item()
		#gen      = fish_info.loc[(fish_info['date']== int(''.join(c for c in f_name[:f_name.find('-')] if c.isdigit()))) & (fish_info['fish-num'] == int(''.join(c for c in f_name[(f_name.find('-')+2):]))), 'genotype'].item()
		genotype = match_names[indx][1][1]
		
		date, fish = f_name.split('-')
		fish = int(re.search(r'\d+', fish).group())
		df.append([date, fish, pd.NA, 'y', dpf, genotype, 'y', 'n', 'y', pd.NA, pd.NA])#['date', 'fish_num', 'clutch-num', 'gravel_reared', 'dpf', 'genotype', 'pre-processed', 'annotated', 'tracked', 'Group', 'notes']
	
	df = pd.DataFrame(df, columns=columns)
	#df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
	
	sys.exit(0)
