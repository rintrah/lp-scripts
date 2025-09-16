"""
Analyze information about bouts. Note that the files of the bouts are in 
MATLAB format. 
"""
import numpy as np
import pandas as pd 

import os
import glob
import ntpath
import sys 

import math

from typing import List, Tuple
from datetime import date
import locale 

import copy

from IPython import get_ipython

import re 
import pdb
from matplotlib import cm, colors
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('../large-scale-analysis/figures.mplstyle')

from itertools import groupby
from operator import itemgetter

from functools import partial 

from scipy.signal import resample, find_peaks
from findpeaks import findpeaks
from scipy.stats import zscore, ks_2samp, ttest_ind, sem, norm, kruskal,  mannwhitneyu, wilcoxon

from scipy.signal import savgol_filter

from scipy.io import loadmat, savemat 

from scipy.interpolate import make_splprep, interp1d
from scipy.signal import resample

from decimal import Decimal

# Matching names.
match_names = [('2021-06-08-14-54-59', ('20210608-f1', 'WT')), ('2021-06-25-15-21-37' ,('20210625-f2', 'HOM')), ('2021-07-06-13-48-55', ('20210706-f1', 'WT')), ('2021-07-13-12-49-43', ('20210713-f1', 'WT')),
('2021-07-16-11-32-30', ('20210716-f1', 'WT')), ('2021-08-27-11-46-32' ,('20210827-f1', 'HOM')), ('2021-09-21-11-37-35', ('20210921-f1', 'WT')), ('2021-10-13-10-44-34', ('20211013-f1', 'HOM')),
('2021-11-18-14-06-44', ('20211118-f11', 'WT')), ('2022-01-20-11-21-57' ,('20220120-f10', 'HOM')), ('2022-02-08-10-21-23', ('20220208-f10', 'WT')), ('2022-02-08-12-42-27', ('20220208-f11', 'WT')),
('2022-02-10-11-11-50', ('20220210-f10', 'HOM')), ('2022-02-16-09-46-03' ,('20220216-f10', 'WT')), ('2022-02-18-13-11-27',('20220218-f11', 'WT')), ('2022-02-18-15-06-54', ('20220218-f12', 'WT')),
('2022-03-14-09-07-13', ('20220314-f10', 'HOM')), ('2022-03-14-12-12-26' ,('20220314-f11', 'HOM')), ('2022-03-15-09-29-12',('20220315-f10', 'HOM')), ('2022-03-15-13-49-02', ('20220315-f12', 'HOM')),
('2022-03-16-14-04-55', ('20220316-f12', 'HOM')), ('2022-03-18-11-40-24' ,('20220318-f11', 'HOM')), ('2022-03-18-14-53-12',('20220318-f12', 'HOM')), ('2022-03-21-14-27-34', ('20220321-f11', 'HOM')),
('2022-03-22-14-03-22', ('20220322-f10', 'WT')), ('2022-03-22-11-59-19' ,('20220322-f11', 'WT')), ('2022-03-23-10-11-32',('20220323-f10', 'WT')), ('2022-03-23-12-10-46', ('20220323-f11', 'WT')),
('2022-03-23-14-01-19', ('20220323-f12', 'WT')), ('2022-03-24-09-04-01' ,('20220324-f10', 'WT'))]

def rot_matrix(theta:float)->np.ndarray:
	# We rotate the fish so the top of the tail tip is at the bottom
	# of the reference space.
	# Rotation matrix.
	theta = np.radians(theta)
	c, s  = np.cos(theta), np.sin(theta)
	R     = np.array(((c, -s), (s, c)))
	return R


def rand_jitter(x:int, stdev:float):
	return (x + np.random.randn(1) * stdev).astype('float')

def vectorize_jitter(x:np.ndarray, stdev:float):
	return np.vectorize(rand_jitter)(x, stdev)

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
	
	# Clear all images.
	plt.close('all')
	
	preds_folder = '/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48/video_preds/'

	dest_fld = os.path.join(preds_folder ,'fish-plots');
	if not os.path.exists(dest_fld):
		os.makedirs(dest_fld)
		
		
	fps = 500
	
	files      = []
	for p, d, f in os.walk(preds_folder):
		for file in f:
			if file.endswith('.csv'):
				if file.endswith('_temporal_norm.csv'):
					pass
				else:
					files.append(os.path.join(p, file))

	bouts_folder = preds_folder + 'bouts-data/'
	
	filenames = next(os.walk(bouts_folder))[2]
	
	# Load XLS file with information about all fish.
	xls_file  = [xls for xls in os.listdir(main_folder) if os.path.isfile(os.path.join(main_folder, xls)) and 'zf-lsm' in xls]
	fish_info = pd.read_excel(os.path.join(main_folder, xls_file[0]))
	
	bouts_gen = {'HOM':{'Seconds':np.array([]), 'Frames':np.array([]), 'Events':np.array([])}, \
				'HOM M':{'Seconds':np.array([]), 'Frames':np.array([]), 'Events':np.array([])}, \
				'WT':{'Seconds':np.array([]), 'Frames':np.array([]), 'Events':np.array([])}, \
				'WT M':{'Seconds':np.array([]), 'Frames':np.array([]), 'Events':np.array([])}} 
				
	sec_gen = {'HOM':{'Spontaneous':np.array([]), 'Light Dot':np.array([]), 'Moving Dot':np.array([])}, \
				'WT':{'Spontaneous':np.array([]), 'Light Dot':np.array([]), 'Moving Dot':np.array([])}} 
	
	all_gen = {'HOM':np.array([]), 'WT':np.array([])}
	
	for file in filenames:
		fish_name = file.replace('-bouts.mat', '')
		
		file_data = np.array([x for x in files if fish_name in x]).item()
		
		indx  = np.argwhere(np.array([x[0] for x in match_names]) == fish_name).item()
		bouts = loadmat(os.path.join(bouts_folder, file))['bouts'].flatten()
		
		idd   = np.where(bouts == 1)[0].flatten()
		num_lst = [] 
		for k, g in groupby(enumerate(idd), lambda ix : ix[0] - ix[1]): 
			num_lst.append(list(map(itemgetter(1), g)))
		
		f_name   = match_names[indx][1][0]
		print(f"For fish {f_name} the size of bouts vector is {bouts.size / fps / 60}.")
		
		min_size = bouts.size//500//60
		#pdb.set_trace()
		windows   = np.array_split(np.arange(0, bouts.size), min_size)
		bouts_min = np.zeros((len(windows),))
		
		for i, window in enumerate(windows):
			idd = np.where(bouts[window] == 1)[0].flatten()
			bm  = []
			for k, g in groupby(enumerate(idd), lambda ix : ix[0] - ix[1]): 
				bm.append(list(map(itemgetter(1), g)))
			bouts_min[i] = len(bm)
		
		
		dpf      = fish_info.loc[(fish_info['date']== int(''.join(c for c in f_name[:f_name.find('-')] if c.isdigit()))) & (fish_info['fish-num'] == int(''.join(c for c in f_name[(f_name.find('-')+2):]))), 'dpf'].item()
		genotype = match_names[indx][1][1]
		#print(f"Genotype is {genotype} and dpf is {dpf} for fish {fish_name}.")
		
		
		stim_info = pd.read_csv(os.path.join(main_folder, os.path.join(f_name, 'LaunchFile_' + f_name + '.csv')))
		start_mov = int(list(stim_info['Fish ID'][stim_info[ f_name[f_name.find('-') + 2:]].str.contains('DarkSpotSpeed', na=False)])[0])
		
		# Separate recording sections. 
		sp_pnt  = np.arange(100, 1750).astype('int')
		ld_pnt  = np.arange(1850,start_mov).astype('int')
		mv_pnt  = np.arange(start_mov, 5200).astype('int') # Remove "magic" numbers.
		all_pnt = np.concatenate((np.arange(100, 1750).astype('int'), np.arange(1850, 5200).astype('int')))
		
		
		frames_lst = np.array(list(map(len, num_lst))) 
		
		sp_lst  = [x for x in num_lst if all(np.isin(np.array(x)//fps, sp_pnt))]
		ld_lst  = [x for x in num_lst if all(np.isin(np.array(x)//fps, ld_pnt))]
		mv_lst  = [x for x in num_lst if all(np.isin(np.array(x)//fps, mv_pnt))]
		all_lst = [x for x in num_lst if all(np.isin(np.array(x)//fps, all_pnt))]
		
		sec_gen[genotype]['Spontaneous'], sec_gen[genotype]['Light Dot'], sec_gen[genotype]['Moving Dot']  = np.append(sec_gen[genotype]['Spontaneous'], len(sp_lst)), np.append(sec_gen[genotype]['Light Dot'], len(ld_lst)), np.append(sec_gen[genotype]['Moving Dot'], len(mv_lst))
		all_gen[genotype] = np.append(all_gen[genotype], bouts_min.mean())
		
		if dpf > 6:
			genotype = genotype + ' M'
		bouts_gen[genotype]['Seconds'], bouts_gen[genotype]['Frames'], bouts_gen[genotype]['Events']  = np.append(bouts_gen[genotype]['Seconds'], frames_lst/500), np.append(bouts_gen[genotype]['Frames'], frames_lst), np.append(bouts_gen[genotype]['Events'], bouts_min.mean())
		
		
		# # Load the tail points obtained by using Ligthning-Pose.
		# df           = pd.read_csv(file_data, header=[1, 2], index_col=0)
		# keypoints_arr = np.reshape(df.to_numpy(), [df.shape[0], -1, 3])
		# # xs_tmp = keypoints_arr[:, :, 0]
		# # ys_tmp = keypoints_arr[:, :, 1] 
		# T = keypoints_arr.shape[0]

		# alpha          = np.linspace(0, 1, 100)
		# xs_arr, ys_arr = np.zeros((T, alpha.size)), np.zeros((T, alpha.size))
		
		# R = rot_matrix(50)
		
		# for i in range(T):
			# points   = keypoints_arr[i, :, :2]
			# distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
			# distance = np.insert(distance, 0, 0)/distance[-1]
			
			# # Interpolate the points used in Lightning-Pose using a quadratic interpolator.
			# interpolator        = interp1d(distance, points, kind='quadratic', axis=0)
			# interpolated_points = interpolator(alpha)
			# # Rotate the points. 
			# #interpolated_points = np.matmul(interpolated_points[:, :2], R)
			# xs_arr[i, :], ys_arr[i, :] =  interpolated_points[:, 0],  interpolated_points[:, 1]
			# del interpolator, interpolated_points, distance, points

		# # Make sure that the uppermost part of the tail is centered at (0, 0). 
		# # I think the problem is here.  
		# for i in range(T):
			# xs_arr[i, :] -= xs_arr[i, 0]
			# ys_arr[i, :] -= ys_arr[i, 0]
		
		# for i in range(T):
			# points         = np.array([xs_arr[i, :], ys_arr[i, :]])
			# rotated_points = np.matmul(R, points).T
			# xs_arr[i, :], ys_arr[i, :] = rotated_points[:, 0], rotated_points[:, 1]
			# del points, rotated_points
		
		# for i in range(len(num_lst)):
			# f_bout = num_lst[i]
			# n_tp   = len(f_bout) 
			# cmap   = plt.get_cmap('winter_r', n_tp)
			# fig, ax = plt.subplots(figsize=(8,8))
			# for j, k in enumerate(f_bout):
				# ax.plot(ys_arr[k, :], xs_arr[k, :], c=cmap(j))
			# ax.set_axis_off()
			# plt.show()
	
	for key in bouts_gen['WT'].keys(): 
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.bar([1, 2], [bouts_gen['HOM'][key].mean(), bouts_gen['WT'][key].mean()], facecolor='none', edgecolor='black', linewidth=3)
		ax.scatter(vectorize_jitter(1 * np.ones((bouts_gen['HOM'][key].size, )), 0.1), bouts_gen['HOM'][key], alpha=0.5, color='red')
		ax.scatter(vectorize_jitter(2 * np.ones((bouts_gen['WT'][key].size, )), 0.1), bouts_gen['WT'][key], alpha=0.5, color='blue')
		if key == 'Events':
			ax.set(xlabel="Genotype", ylabel=f"# of bouts per minute", title='younger larvae')
		else:
			ax.set(xlabel="Genotype", ylabel=f"# of {key}", title='younger larvae')
		ax.set_xticks([1, 2])
		ax.set_xticklabels(['fmr1-/-', 'wild type'])
		stats, pvalue = mannwhitneyu(bouts_gen['HOM'][key], bouts_gen['WT'][key])
		bottom, top = ax.get_ylim()
		ax.hlines(y=top , xmin=1, xmax=2, linewidth=2, color='k')
		ax.text(1.5, top + 0.02, f'p={Decimal(pvalue):.2E}', fontsize=12)
		plt.savefig(os.path.join(dest_fld, f"comparison-{key}-between-young-larva.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
		plt.close('all')
	
	for key in bouts_gen['WT'].keys(): 
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.bar([1, 2], [bouts_gen['HOM M'][key].mean(), bouts_gen['WT M'][key].mean()], facecolor='none', edgecolor='black', linewidth=3)
		ax.scatter(vectorize_jitter(1 * np.ones((bouts_gen['HOM M'][key].size, )), 0.1), bouts_gen['HOM M'][key], alpha=0.5, color='red')
		ax.scatter(vectorize_jitter(2 * np.ones((bouts_gen['WT M'][key].size, )), 0.1), bouts_gen['WT M'][key], alpha=0.5, color='blue')
		if key == 'Events':
			ax.set(xlabel="Genotype", ylabel=f"# of bouts per minute", title='older larvae')
		else:
			ax.set(xlabel="Genotype", ylabel=f"# of {key}", title='older larvae')
		ax.set_xticks([1, 2])
		ax.set_xticklabels(['fmr1-/-', 'wild type'])
		stats, pvalue = mannwhitneyu(bouts_gen['HOM M'][key], bouts_gen['WT M'][key])
		bottom, top = ax.get_ylim()
		ax.hlines(y=top, xmin=1, xmax=2, linewidth=2, color='k')
		ax.text(1.5, top + 0.02, f'p={Decimal(pvalue):.2E}', fontsize=12)
		plt.savefig(os.path.join(dest_fld, f"comparison-{key}-between-older-larva.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
		plt.close('all')
		
	for key in sec_gen['WT'].keys(): 
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.bar([1, 2], [sec_gen['HOM'][key].mean(), sec_gen['WT'][key].mean()], facecolor='none', edgecolor='black', linewidth=3)
		ax.scatter(vectorize_jitter(1 * np.ones((sec_gen['HOM'][key].size, )), 0.1), sec_gen['HOM'][key], alpha=0.5, color='red')
		ax.scatter(vectorize_jitter(2 * np.ones((sec_gen['WT'][key].size, )), 0.1), sec_gen['WT'][key], alpha=0.5, color='blue')
		ax.set(xlabel="Genotype", ylabel=f"# of {key} events")
		ax.set_xticks([1, 2])
		ax.set_xticklabels(['fmr1-/-', 'wild type'])
		stats, pvalue = mannwhitneyu(sec_gen['HOM'][key], sec_gen['WT'][key])
		bottom, top = ax.get_ylim()
		ax.hlines(y=top, xmin=1, xmax=2, linewidth=2, color='k')
		ax.text(1.5, top + 0.02, f'p={Decimal(pvalue):.2E}', fontsize=12)
		print(f"For section {key} the p-value is {pvalue}.")
		plt.savefig(os.path.join(dest_fld, f"comparison-num-events-during-{key}-larva.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
		plt.close('all')
		
	#for key in sec_gen['WT'].keys(): 
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.bar([1, 2], [all_gen['HOM'].mean(), all_gen['WT'].mean()], facecolor='none', edgecolor='black', linewidth=3)
	ax.scatter(vectorize_jitter(1 * np.ones((all_gen['HOM'].size, )), 0.1), all_gen['HOM'], alpha=0.5, color='red')
	ax.scatter(vectorize_jitter(2 * np.ones((all_gen['WT'].size, )), 0.1), all_gen['WT'], alpha=0.5, color='blue')
	ax.set(xlabel="Genotype", ylabel=f"# of bouts per minute")
	ax.set_xticks([1, 2])
	ax.set_xticklabels(['fmr1-/-', 'wild type'])
	stats, pvalue = mannwhitneyu(all_gen['HOM'], all_gen['WT'])
	bottom, top = ax.get_ylim()
	ax.hlines(y=top, xmin=1, xmax=2, linewidth=2, color='k')
	ax.text(1.5, top + 0.02, f'p={Decimal(pvalue):.2E}', fontsize=12)
	print(f"For section {key} the p-value is {pvalue}.")
	plt.savefig(os.path.join(dest_fld, f"comparison-num-bouts-min-during-recording-larva.png")	, dpi=300, format='png', bbox_inches="tight", transparent=False)
	plt.close('all')
		
	pdb.set_trace()
	
	sys.exit(0)
