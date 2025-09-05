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

from scipy.signal import resample

from scipy.signal import savgol_filter

if __name__ == '__main__':
	preds_folder = '/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48/video_preds/'

	dest_fld = os.path.join(preds_folder ,'fish-plots');
	if not os.path.exists(dest_fld):
		os.makedirs(dest_fld)

	df = pd.read_hdf(os.path.join(dest_fld, 'fish-tail-tip-angle-info.h5'))
	
	# ['Fish',  'Genotype',  'Angle']
	keys = df['Fish'].unique()
	
	fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
	for key in keys:
		values   = df[df['Fish'] == key]['Angle'].values
		genotype = df[df['Fish'] == key]['Genotype'].unique()
		if np.nanmin(values) < -100:
			print(f"Fish {key} has outlier values.")
			pass
		else:
			if genotype == 'WT':
				axes[0].hist(np.abs(values), bins='fd', histtype='step', alpha=0.3)
			else:
				axes[1].hist(np.abs(values), bins='fd', histtype='step', alpha=0.3)
	axes[0].set(ylabel='# Count')
	axes[1].set(xlabel='Angle', ylabel='# Count')
	plt.show()
	
	pdb.set_trace()
	sys.exit(0)
