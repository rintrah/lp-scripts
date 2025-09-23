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

import seaborn as sns 

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
	
def plot_sec_angle(df:pd.DataFrame, my_palette:dict, dest_fld:str, age:str):
	fig, ax = plt.subplots(figsize=(8, 8))
	sns.kdeplot(data=df, x="Seconds", y="Angle", hue="Genotype", fill=True, palette=my_palette, alpha=0.3)
	plt.savefig(os.path.join(dest_fld, f"scatter-sec-angle-{age}-larvae.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
	plt.close('all')

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
		
	# Parameters of the recording.
	fps     = 500
	sec_min = 60 
	
	files      = []
	for p, d, f in os.walk(preds_folder):
		for file in f:
			if file.endswith('.csv'):
				if file.endswith('_temporal_norm.csv'):
					pass
				else:
					files.append(os.path.join(p, file))

	bouts_folder  = preds_folder + 'bouts-data/'
	degree_folder = preds_folder + 'degree-data/'
	
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
	
	times_gen = {'HOM':np.array([]), 'WT':np.array([])}
	
	#stim_bouts = {'HOM':{'in':np.array([]), 'out':np.array([])}, 'WT':{'in':np.array([]), 'out':np.array([])}}
	stim_bouts = {'HOM':{'in':[], 'out':[]}, 'WT':{'in':[], 'out':[]}}
	
	
	df_bouts = [] 
	test     = [] 
	for file in filenames:
		fish_name = file.replace('-bouts.mat', '')
		
		# There is a difference between the name of the Video file and the one of the Calcium file.
		# Because of this, we look for the Calcium file that matches the Video files.
		indx  = np.argwhere(np.array([x[0] for x in match_names]) == fish_name).item()
		f_name   = match_names[indx][1][0]

		file_data = np.array([x for x in files if fish_name in x]).item()
		dpf      = fish_info.loc[(fish_info['date']== int(''.join(c for c in f_name[:f_name.find('-')] if c.isdigit()))) & (fish_info['fish-num'] == int(''.join(c for c in f_name[(f_name.find('-')+2):]))), 'dpf'].item()
		genotype = match_names[indx][1][1]
		#print(f"Genotype is {genotype} and dpf is {dpf} for fish {fish_name}.")
		


		
		# Bouts is a binary array. 
		bouts = loadmat(os.path.join(bouts_folder, file))['bouts'].flatten()
		# Check the size in frames of the bout vector.
		print(f"For fish {f_name} the size of bouts vector is {bouts.size / fps / 60}.")
		

		
		# We want to compare the length of the bouts with the maximum tail angle.
		deg_data = loadmat(os.path.join(degree_folder, fish_name + '-tail-deg.mat'))['tail']
		deg_data = np.abs(deg_data[:, -1])
		
		# Here we find consecutive ones.
		idd     = np.where(bouts == 1)[0].flatten()
		num_lst = [] 
		for k, g in groupby(enumerate(idd), lambda ix : ix[0] - ix[1]): 
			tmp_range = list(map(itemgetter(1), g))
			if np.max(deg_data[tmp_range]) < 200:
				num_lst.append(tmp_range)
				test.append([f_name, dpf, genotype, len(num_lst[-1])/fps, np.max(deg_data[num_lst[-1]])])
			
			# if np.max(deg_data[tmp_range]) > 100:
				# deg_data = loadmat(os.path.join(degree_folder, fish_name + '-tail-deg.mat'))['tail']
				# cmap = plt.get_cmap('coolwarm', 100)
				# norm = colors.Normalize(tmp_range[0], tmp_range[-1])
				# for i in tmp_range: 
					# plt.plot(deg_data[i, :], c=cmap(norm(i)), alpha=0.3)
				# plt.show()
			
		min_size = bouts.size//fps//sec_min
		
		windows   = np.array_split(np.arange(0, bouts.size), min_size)
		bouts_min = np.zeros((len(windows),))
		
		time_btw_bouts = [] 
		for b in range(len(num_lst) - 1):
			time_btw_bouts.append(num_lst[b+1][0] - num_lst[b][-1])
		
		for i, window in enumerate(windows):
			idd = np.where(bouts[window] == 1)[0].flatten()
			bm  = []
			for k, g in groupby(enumerate(idd), lambda ix : ix[0] - ix[1]): 
				bm.append(list(map(itemgetter(1), g)))
			bouts_min[i] = len(bm)
		
		
		idd = np.where(bouts == 0)[0].flatten()
		n_b = [] 
		for k, g in groupby(enumerate(idd), lambda ix : ix[0] - ix[1]): 
			n_b.append(list(map(itemgetter(1), g)))

		
		stim_info = pd.read_csv(os.path.join(main_folder, os.path.join(f_name, 'LaunchFile_' + f_name + '.csv')))
		start_mov = int(list(stim_info['Fish ID'][stim_info[f_name[f_name.find('-') + 2:]].str.contains('DarkSpotSpeed', na=False)])[0])
		
		lst_names    = list(stim_info[f_name[f_name.find('-') + 2:]].str.contains('DarkSpotLoc', na=False))
		unique_names = list(map(lambda x : x [:x.find('.')], np.unique(stim_info[f_name[f_name.find('-') + 2:]][np.where(lst_names)[0]])))
		dict_stim = {}
		for name in unique_names:
			dict_stim[name] = list(map(int, list(stim_info['Fish ID'][stim_info[f_name[f_name.find('-') + 2:]].str.contains(name, na=False)])))
		
		
		b_stim   = {}
		for i in range(len(dict_stim.keys())): 
			b_stim.update({str(i): []})
			
		b_n_stim = {} 
		for i in range(len(dict_stim.keys())): 
			b_n_stim.update({str(i): []})
			
		in_stim     = np.zeros((len(dict_stim.keys()), 10))
		not_in_stim = np.zeros((len(dict_stim.keys()), 10))
		
		for i, key in enumerate(dict_stim.keys()):
			for j, start in enumerate(dict_stim[key]):
				stim_range   = np.arange(start * fps, (start + 3) * fps)
				n_stim_range = np.arange((start + 3) * fps, (start + 6) * fps)
				for k in range(len(num_lst)):
					if num_lst[k][0] in stim_range:
						b_stim[str(i)].append(np.abs(num_lst[k][0] - stim_range[0])/fps)
						in_stim[i][j] += 1
					elif num_lst[k][0] in n_stim_range:
						b_n_stim[str(i)].append(np.abs(num_lst[k][0] - n_stim_range[0])/fps)
						not_in_stim[i][j] += 1
		
		# stim_bouts[genotype]['in']  = np.append(stim_bouts[genotype]['in'], in_stim.sum(axis=1))
		# stim_bouts[genotype]['out'] = np.append(stim_bouts[genotype]['out'], not_in_stim.sum(axis=1))
		
		stim_bouts[genotype]['in'].append(in_stim.sum(axis=1))
		stim_bouts[genotype]['out'].append(not_in_stim.sum(axis=1))
		
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
		times_gen[genotype]= np.append(times_gen[genotype], time_btw_bouts)
		
		for frame in frames_lst: 
			df_bouts.append([f_name, genotype, dpf, frame/fps])
		
		sp_cnt, ld_cnt, mv_cnt = 0, 0, 0 
		
		for x in num_lst:
			if np.isin(np.array(x[0]), np.arange(100 * fps, 1750 * fps)):
				sp_cnt += 1
			elif np.isin(np.array(x[0]), np.arange(1850 * fps, start_mov * fps)):
				ld_cnt += 1
			elif np.isin(np.array(x[0]), np.arange(start_mov * fps, 5200 * fps)):
				mv_cnt += 1
				
		print(f"{f_name} has {sp_cnt} during spontaneous activity, {ld_cnt} bouts during light dot, and {mv_cnt} bouts during moving dot.")

		if dpf > 6:
			genotype = genotype + ' M'
		bouts_gen[genotype]['Seconds'], bouts_gen[genotype]['Frames'], bouts_gen[genotype]['Events']  = np.append(bouts_gen[genotype]['Seconds'], frames_lst/fps), np.append(bouts_gen[genotype]['Frames'], frames_lst), np.append(bouts_gen[genotype]['Events'], bouts_min.mean())
	
	df_info = pd.DataFrame(test, columns=['Subject', 'Dpf', 'Genotype', 'Seconds', 'Angle'])
	
	m_df = df_info[df_info['Dpf'] > 6]
	my_palette = {'HOM':'#D6604D', 'WT': '#4393C3'}
	plot_sec_angle(m_df, my_palette, dest_fld, 'mature')
	
	y_df = df_info[df_info['Dpf'] < 7]
	my_palette = {'HOM': '#F4A582', 'WT': '#92C5DE'}
	plot_sec_angle(y_df, my_palette, dest_fld, 'young')
	
	df_bouts = pd.DataFrame(df_bouts, columns=['Subject', 'Genotype', 'Dpf', 'Seconds'])
	
	df_bouts['dummy'] = 0
	my_palette = {'HOM': '#B2182B', 'WT': '#053061'}
	fig, ax = plt.subplots()
	sns.violinplot(data=df_bouts, x='dummy', y='Seconds', split=True, hue='Genotype', ax=ax, palette=my_palette)
	#ax.axes.get_xaxis().set_visible(False)
	ax.set_xticks([])
	ax.legend(frameon=False)
	ax.set_xlabel('Density')
	ax.set(ylabel='Duration of bouts [s]')
	plt.savefig(os.path.join(dest_fld, f"violinplot-comparison-genotypes-duration-bouts.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
	plt.close('all')
	
	
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
	
	
	my_palette = {'HOM': '#F4A582' ,'HOM M': '#D6604D', 'WT': '#92C5DE', 'WT M': '#4393C3'}
	
	df = [] 
	for key in bouts_gen.keys():
		for i in range(bouts_gen[key]['Seconds'].size):
			df.append([key, bouts_gen[key]['Seconds'][i]])
	
	df = pd.DataFrame(df, columns=['Genotype', 'Seconds'])
	
	dummy_df = df[np.logical_or(df['Genotype']=='HOM', df['Genotype'] == 'WT')]
	dummy_df['dummy'] = 0
	fig, ax = plt.subplots()
	sns.violinplot(data=dummy_df, x='dummy', y='Seconds', split=True, hue='Genotype', ax=ax, palette=my_palette)
	#ax.axes.get_xaxis().set_visible(False)
	ax.set_xticks([])
	ax.legend(frameon=False)
	ax.set_xlabel('Density')
	ax.set(ylabel='Duration of bouts [s]')
	plt.savefig(os.path.join(dest_fld, f"violinplot-comparison-genotypes-young-duration-bouts.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
	plt.close('all')
	
	# Violin plots of age groups. 
	
	dummy_df = df[np.logical_or(df['Genotype']=='HOM M', df['Genotype'] == 'WT M')]
	dummy_df['dummy'] = 0
	fig, ax = plt.subplots()
	sns.violinplot(data=dummy_df, x='dummy', y='Seconds', split=True, hue='Genotype', ax=ax, palette=my_palette)
	#ax.axes.get_xaxis().set_visible(False)
	ax.set_xticks([])
	ax.legend(frameon=False)
	ax.set_xlabel('Density')
	ax.set(ylabel='Duration of bouts [s]')
	plt.savefig(os.path.join(dest_fld, f"violinplot-comparison-genotypes-older-duration-bouts.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
	plt.close('all')
	
	hom_v, wt_v = np.vstack(stim_bouts['HOM']['in']).mean(axis=0), np.vstack(stim_bouts['WT']['in']).mean(axis=0)
	
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.bar([1, 2], [hom_v.mean(), wt_v.mean()], facecolor='none', edgecolor='black', linewidth=3)
	ax.scatter(vectorize_jitter(1 * np.ones((hom_v.size, )), 0.1), hom_v, alpha=0.5, color='red')
	ax.scatter(vectorize_jitter(2 * np.ones((wt_v.size, )), 0.1), wt_v, alpha=0.5, color='blue')
	ax.set(xlabel="Genotype", ylabel=f"# bouts after stimulation")
	ax.set_xticks([1, 2])
	ax.set_xticklabels(['fmr1-/-', 'wild type'])
	stats, pvalue = mannwhitneyu(hom_v, wt_v)
	bottom, top = ax.get_ylim()
	ax.hlines(y=top, xmin=1, xmax=2, linewidth=2, color='k')
	ax.text(1.5, top + 0.02, f'p={Decimal(pvalue):.2E}', fontsize=12)
	print(f"For the average bouts for each stimulus the p-value is {pvalue}.")
	plt.savefig(os.path.join(dest_fld, f"comparison-num-bouts-after-stimulation-larva.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
	plt.close('all')
	
	
	pdb.set_trace()
	
	sys.exit(0)
