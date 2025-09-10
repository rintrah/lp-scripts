"""
Analyze tail angles information which is saved in a Panda DataFrame.
"""
import numpy as np
import pandas as pd 

import os
import glob
import ntpath
import sys 

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

from functools import partial 

from scipy.signal import resample, find_peaks
from findpeaks import findpeaks

from scipy.signal import savgol_filter

if __name__ == '__main__':
	preds_folder = '/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48/video_preds/'

	dest_fld = os.path.join(preds_folder ,'fish-plots');
	if not os.path.exists(dest_fld):
		os.makedirs(dest_fld)

	df = pd.read_hdf(os.path.join(dest_fld, 'fish-tail-tip-angle-info.h5'))
	
	# ['Fish',  'Genotype',  'Angle']
	keys = df['Fish'].unique()
	
	# fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
	# for key in keys:
		# if key != "20220322-f11":
			# values   = df[df['Fish'] == key]['Angle'].values
			# genotype = df[df['Fish'] == key]['Genotype'].unique()
			# if np.nanmin(values) < -100:
				# print(f"Fish {key} has outlier values.")
				# pass
			# else:
				# if genotype == 'WT':
					# axes[0].hist(np.abs(values), bins='fd', histtype='step', alpha=0.3)
				# else:
					# axes[1].hist(np.abs(values), bins='fd', histtype='step', alpha=0.3)
	# axes[0].set(ylabel='# Count')
	# axes[1].set(xlabel='Angle', ylabel='# Count')
	# plt.show()
	
	for key in keys:
		if key != "20220322-f11":
			values   = df[df['Fish'] == key]['Angle'].values
			genotype = df[df['Fish'] == key]['Genotype'].unique()
			func = np.abs(values)
			n_func, bins, patches = plt.hist(func, bins='fd')
			# fp      = findpeaks(method='peakdetect',lookahead=1);
			# results = fp.fit(n_func);
			# peaks   = np.where(results['df']['peak'] == True)[0].flatten();
			sel_height = np.nanmean(n_func)
			peaks, _   = find_peaks(n_func,  height=sel_height)
			
			centers = np.round(bins[:-1] + (bins[1:] - bins[:-1])/2, 1)
			plt.close('all')
			fig, ax = plt.subplots(figsize=(16, 8))
			ax.plot(np.arange(0, n_func.size), n_func, color='#2166ac', lw=2, alpha=0.5)
			ax.scatter(peaks, n_func[peaks], c='#b2182b', marker='o', )
			ax.set_xticks(np.arange(0, centers.size)[::20])
			ax.set_xticklabels(centers[::20])
			ax.set(xlabel=r"Tail-tip Angle ($\circ$)", ylabel="Count")
			ax.set_title(f"Fish {key}")
			ax.autoscale(enable=True, axis='x', tight=True)
			plt.savefig(os.path.join(dest_fld, f"{key}-histogram-tail-tip-angles.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
			plt.close('all')
	# pk_ind  = np.argsort(f(x_arr)[peaks])[::-1]    
	# peaks   = peaks[pk_ind]
	
	sys.exit(0)
