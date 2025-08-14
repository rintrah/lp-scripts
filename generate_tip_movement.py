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

from scipy.interpolate import make_splprep, interp1d
from scipy.signal import resample

from scipy.signal import savgol_filter

def plot_trajectory(files:str, fps:int, dest_fld:str):
	sec = 60 
	for file_data in files:
		folder, file = ntpath.split(file_data)
		fish_name    = file.replace('.csv', '')
		df           = pd.read_csv(file_data, header=[1, 2], index_col=0)
		keypoints_arr = np.reshape(df.to_numpy(), [df.shape[0], -1, 3])
		# xs_tmp = keypoints_arr[:, :, 0]
		# ys_tmp = keypoints_arr[:, :, 1] 
		T = keypoints_arr.shape[0]
		
		alpha = np.linspace(0, 1, 100)
		xs_arr, ys_arr = np.zeros((T, alpha.size)), np.zeros((T, alpha.size))
		
		for i in range(T):
			points   = keypoints_arr[i, :, :2]
			distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
			distance = np.insert(distance, 0, 0)/distance[-1]
			
			interpolator        = interp1d(distance, points, kind='quadratic', axis=0)
			interpolated_points = interpolator(alpha)
			xs_arr[i, :], ys_arr[i, :] =  interpolated_points[:, 0],  interpolated_points[:, 1]
			
			# fig, ax = plt.subplots(figsize=(8, 8))
			# ax.plot(xs_arr[i, :], ys_arr[i, :], '-')
			# ax.plot(points[:, 0], points[:, 1], '.')
			# plt.show()
			
		# for i in range(T):
			# xs_arr[i, :] -= xs_arr[i, 0]
			# ys_arr[i, :] -= ys_arr[i, 0]
			
		theta = np.radians(30)
		c, s = np.cos(theta), np.sin(theta)
		R = np.array(((c, -s), (s, c)))
		xs_series = xs_arr[:, -1]
		ys_series = ys_arr[:, -1]
		
		xs_ds = resample(xs_series, xs_series.size // fps)
		ys_ds = resample(ys_series, ys_series.size // fps)
		
		xs_matrix = np.zeros((xs_series.size // fps, xs_arr.shape[1]))
		ys_matrix = np.zeros((ys_series.size // fps, ys_arr.shape[1]))
		
		for i in range(xs_arr.shape[1]):
			xs_matrix[:, i] = resample(xs_arr[:, i], xs_series.size // fps)
			ys_matrix[:, i] = resample(ys_arr[:, i], ys_series.size // fps)
		
		dx = -np.diff(xs_matrix, 1, axis=1)
		dy = -np.diff(ys_matrix, 1, axis=1)
		rad_ang   = np.arctan2(dy, dx)
		deg_matrix = np.rad2deg(rad_ang)
		
		for i in np.arange(1, deg_matrix.shape[1] + 1):
			dtheta = np.array([deg_matrix[:, i] - deg_matrix[:, i-1], 
						deg_matrix[:, i] - deg_matrix[:, i-1] + 360,
						deg_matrix[:, i] - deg_matrix[:, i-1] - 360]).T
			indx = np.argmin(np.abs(dtheta), axis= 1)
			deg_matrix[:, i] = deg_matrix[:, i - 1] + np.take_along_axis(dtheta, indx[:, None], axis=1).flatten()

		deg_ang = deg_matrix[:, -1] #np.rad2deg(rad_ang[:, -1])
		
		y = savgol_filter(ys_ds, window_length=sec, polyorder=0)
		#y = savgol_filter(ys_matrix[:, 0], window_length=sec, polyorder=0)
		
		fig = plt.figure(figsize=(8, 8))
		gs  = fig.add_gridspec(3,3)
		ax1 = fig.add_subplot(gs[0, :])
		ax2 = fig.add_subplot(gs[1:, :2])
		ax3 = fig.add_subplot(gs[1:, 2])
		
		#ax1.plot(np.arange(0, y.size)/sec, ys_ds - y, color='#7570b3', linewidth=2)
		ax1.plot(np.arange(0, xs_ds.size)/sec, deg_ang, color='#7570b3', linewidth=2)
		ax1.set(xlabel="Time [s]", ylabel=r"Tail angle ($\circ$)")
		for i in range(xs_matrix.shape[0]):
			ax2.plot(xs_matrix[i, :], ys_matrix[i, :], color='#666666', alpha=0.2)
		ax2.scatter(xs_ds, ys_ds, c='#7570b3')
		ax2.yaxis.set_inverted(True)
		ax2.set_axis_off() 
		ax3.hist(deg_ang, bins='fd', edgecolor='none', facecolor='#7570b3')
		ax3.set(xlabel=r"Tail angle ($\circ$)", ylabel="# count")
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
