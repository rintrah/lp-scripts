"""
Script that generates times series of the tip of the 
tail. The tail tracking was done using Lightning Pose.
"""
import numpy as np
import pandas as pd 

import os
import glob
import ntpath
import sys 

import h5py
import json 

from typing import List, Tuple
from datetime import date
import locale 

import copy

import pdb

from IPython import get_ipython
import pickle

import re 
import pdb
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
plt.style.use('../large-scale-analysis/figures.mplstyle')

from functools import partial 

from scipy.interpolate import make_splprep
from scipy.signal import resample

from scipy.signal import savgol_filter

def plot_trajectory(files:str, fps:int, dest_fld:str):
	sec = 60 
	for file_data in files:
		folder, file = ntpath.split(file_data)
		fish_name    = file.replace('.csv', '')
		df = pd.read_csv(file_data, header=[1, 2], index_col=0)
		keypoints_arr = np.reshape(df.to_numpy(), [df.shape[0], -1, 3])
		xs_arr = keypoints_arr[:, :, 0]
		ys_arr = keypoints_arr[:, :, 1] 
		
		xs_series = xs_arr[:, -1]
		ys_series = ys_arr[:, -1]
		
		pdb.set_trace()
		
		xs_ds = resample(xs_series, xs_series.size // fps)
		ys_ds = resample(ys_series, ys_series.size // fps)
		
		xs_matrix = np.zeros((xs_series.size // fps, xs_arr.shape[1]))
		ys_matrix = np.zeros((ys_series.size // fps, ys_arr.shape[1]))
		
		for i in range(xs_arr.shape[1]):
			xs_matrix[:, i] = resample(xs_arr[:, i], xs_series.size // fps)
			ys_matrix[:, i] = resample(ys_arr[:, i], ys_series.size // fps)
		
		y = savgol_filter(ys_ds, window_length=sec, polyorder=0)
		#y = savgol_filter(ys_matrix[:, 0], window_length=sec, polyorder=0)
		
		fig = plt.figure(figsize=(8, 8))
		gs  = fig.add_gridspec(3,3)
		ax1 = fig.add_subplot(gs[0, :])
		ax2 = fig.add_subplot(gs[1:, :2])
		ax3 = fig.add_subplot(gs[1:, 2])

		ax1.plot(np.arange(0, y.size)/sec, ys_ds - y, color='#7570b3', linewidth=2)
		ax1.set(xlabel="Time [s]", ylabel="Tail y-position")
		for i in range(xs_matrix.shape[0]):
			ax2.plot(xs_matrix[i, :], ys_matrix[i, :], color='#666666', alpha=0.2)
		ax2.scatter(xs_ds, ys_ds, c='#7570b3')
		ax2.yaxis.set_inverted(True)
		ax2.set_axis_off() 
		ax3.hist(ys_ds - y, bins='fd', edgecolor='none', facecolor='#7570b3')
		ax3.set(xlabel='Tail y-position', ylabel="# count")
		fig.tight_layout()
		plt.suptitle(f"Fish {fish_name}")
		plt.savefig(os.path.join(dest_fld, f"fish-{fish_name}-tail-tracking-results.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
		plt.close('all')
		
		# plt.scatter(xs_ds, ys_ds - y, s=5)
		# plt.gca().invert_yaxis()
		# plt.show()
		# pdb.set_trace()
	
	return 


if __name__ == '__main__':
	preds_folder = '/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48/video_preds/'
	fps = 500
	
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
		
	plot_trajectory(files, fps, dest_fld)
	
	sys.exit(0)
