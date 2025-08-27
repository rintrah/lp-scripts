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

from IPython import get_ipython
import pickle

import re 
import pdb
from matplotlib import cm, colors
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('../large-scale-analysis/figures.mplstyle')

from functools import partial 

from scipy.interpolate import make_splprep, interp1d
from scipy.signal import resample

from scipy.signal import savgol_filter

from scipy.io import savemat # For saving as mat file. 

# Matching names.
match_names = [('2021-06-08-14-54-59', ('20210608-f1', 'WT')), ('2021-06-25-15-21-37' ,('20210625-f2', 'HOM')), ('2021-07-06-13-48-55', ('20210706-f1', 'WT')), ('2021-07-13-12-49-43', ('20210713-f1', 'WT')),
('2021-07-16-11-32-30', ('20210716-f1', 'WT')), ('2021-08-27-11-46-32' ,('20210827-f11', 'HOM')), ('2021-09-21-11-37-35', ('20210921-f1', 'WT')), ('2021-10-13-10-44-34', ('20211013-f1', 'HOM')),
('2021-11-18-14-06-44', ('20211118-f11', 'WT')), ('2022-01-20-11-21-57' ,('20220120-f10', 'HOM')), ('2022-02-08-10-21-23', ('20220208-f10', 'WT')), ('2022-02-08-12-42-27', ('20220208-f11', 'WT')),
('2022-02-10-11-11-50', ('20220210-f10', 'HOM')), ('2022-02-16-09-46-03' ,('20220216-f10', 'WT')), ('2022-02-18-13-11-27',('20220218-f11', 'WT')), ('2022-02-18-15-06-54', ('20220218-f12', 'WT')),
('2022-03-14-09-07-13', ('20220314-f10', 'HOM')), ('2022-03-14-12-12-26' ,('20220314-f11', 'HOM')), ('2022-03-15-09-29-12',('20220315-f10', 'HOM')), ('2022-03-15-13-49-02', ('20220315-f12', 'HOM')),
('2022-03-16-14-04-55', ('20220316-f12', 'HOM')), ('2022-03-18-11-40-24' ,('20220318-f11', 'HOM')), ('2022-03-18-14-53-12',('20220318-f12', 'HOM')), ('2022-03-21-14-27-34', ('20220321-f11', 'HOM')),
('2022-03-22-14-03-22', ('20220322-f10', 'WT')), ('2022-03-22-11-59-19' ,('20220322-f11', 'WT')), ('2022-03-23-10-11-32',('20220323-f10', 'WT')), ('2022-03-23-12-10-46', ('20220323-f11', 'WT')),
('2022-03-23-14-01-19', ('20220323-f12', 'WT')), ('2022-03-24-09-04-01' ,('20220324-f10', 'WT'))]

def wrap180(angle:float)->float:
	return ((angle + 180) % 360) - 180
	
def rot_matrix(theta:float)->np.ndarray:
	# We rotate the fish so the top of the tail tip is at the bottom
	# of the reference space.
	# Rotation matrix.
	theta = np.radians(theta)
	c, s  = np.cos(theta), np.sin(theta)
	R     = np.array(((c, -s), (s, c)))
	return R

def plot_trajectory(files:list, fps:int, dest_fld:str)->pd.DataFrame:
	"""
	Plot the tail positions during the recording and returns a dictionary
	with the number of tail movements of each larva.
	Parameters
	----------
	files: list,
		List of the full paths of the larva data. 
	fps: int, 
		The frame per seconds of the recording.
	dest_fld: str,
		Full path where the images will be saved.
	Returns 
	-------
	df_mov: pd.DataFrame,
		DataFrame with the total number of movements during the recording
		of each larva.
	"""
	sec = 60 
	
	df_mov = [] 
	
	for file_data in files:
		folder, file = ntpath.split(file_data)
		fish_name    = file.replace('.csv', '')
		indx = np.argwhere(np.array([x[0] for x in match_names]) == fish_name).item()
		fish_id, genotype = match_names[indx][1]
		
		# Load the tail points obtained by using Ligthning-Pose.
		df           = pd.read_csv(file_data, header=[1, 2], index_col=0)
		keypoints_arr = np.reshape(df.to_numpy(), [df.shape[0], -1, 3])
		# xs_tmp = keypoints_arr[:, :, 0]
		# ys_tmp = keypoints_arr[:, :, 1] 
		T = keypoints_arr.shape[0]

		alpha          = np.linspace(0, 1, 100)
		xs_arr, ys_arr = np.zeros((T, alpha.size)), np.zeros((T, alpha.size))
		
		R = rot_matrix(50)
		
		for i in range(T):
			points   = keypoints_arr[i, :, :2]
			distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
			distance = np.insert(distance, 0, 0)/distance[-1]
			
			# Interpolate the points used in Lightning-Pose using a quadratic interpolator.
			interpolator        = interp1d(distance, points, kind='quadratic', axis=0)
			interpolated_points = interpolator(alpha)
			# Rotate the points. 
			#interpolated_points = np.matmul(interpolated_points[:, :2], R)
			xs_arr[i, :], ys_arr[i, :] =  interpolated_points[:, 0],  interpolated_points[:, 1]
			del interpolator, interpolated_points, distance, points

		# Make sure that the uppermost part of the tail is centered at (0, 0). 
		# I think the problem is here.  
		for i in range(T):
			xs_arr[i, :] -= xs_arr[i, 0]
			ys_arr[i, :] -= ys_arr[i, 0]
		
		for i in range(T):
			points         = np.array([xs_arr[i, :], ys_arr[i, :]])
			rotated_points = np.matmul(R, points).T
			xs_arr[i, :], ys_arr[i, :] = rotated_points[:, 0], rotated_points[:, 1]
			del points, rotated_points


		# fig, ax = plt.subplots(figsize=(8, 8))
		# for i in range(T):
			# ax.plot(xs_arr[i, 0], ys_arr[i, 0], '-', alpha=0.5)
		# plt.show()
			
		dx         = -np.diff(xs_arr, 1, axis=1)
		dy         = -np.diff(ys_arr, 1, axis=1)
		rad_ang    = np.arctan2(dy, dx)
		deg_matrix = np.rad2deg(rad_ang)
		
		vfunc = np.vectorize(wrap180)
		for i in range(deg_matrix.shape[1]):
			deg_matrix[:, i] = vfunc(deg_matrix[:, i])
		
		#pdb.set_trace()
		for i in np.arange(1, deg_matrix.shape[1]):
			dtheta = np.array([deg_matrix[:, i] - deg_matrix[:, i-1], 
						deg_matrix[:, i] - deg_matrix[:, i-1] + 360,
						deg_matrix[:, i] - deg_matrix[:, i-1] - 360]).T
			indx = np.argmin(np.abs(dtheta), axis= 1)
			deg_matrix[:, i] = deg_matrix[:, i - 1] + np.take_along_axis(dtheta, indx[:, None], axis=1).flatten()
		
		
		xs_series = xs_arr[:, -1]
		ys_series = ys_arr[:, -1]
		
		xs_ds   = resample(xs_series, xs_series.size // fps)
		ys_ds   = resample(ys_series, ys_series.size // fps)
		deg_ang = resample(deg_matrix[:, -1], deg_matrix[:, -1].size // fps) #np.rad2deg(rad_ang[:, -1])
		
		xs_matrix = np.zeros((xs_series.size // fps, xs_arr.shape[1]))
		ys_matrix = np.zeros((ys_series.size // fps, ys_arr.shape[1]))
		
		
		for i in range(xs_arr.shape[1]):
			xs_matrix[:, i] = resample(xs_arr[:, i], xs_series.size // fps)
			ys_matrix[:, i] = resample(ys_arr[:, i], ys_series.size // fps)
		
		mean, sd  = np.nanmean(deg_ang), np.nanstd(deg_ang)
		
		movs = np.union1d(np.where(deg_ang > mean + sd)[0].flatten(), np.where(deg_ang < mean - sd)[0].flatten())
		
		df_mov.append([fish_id, genotype, movs])
		
		#val_range = np.linspace(np.nanmin(deg_ang), np.nanmax(deg_ang), 100)
		cmap      = plt.get_cmap('coolwarm', 100)
		norm      = colors.Normalize(np.nanmin(deg_ang), np.nanmax(deg_ang))
		
		# Running average.
		#y = savgol_filter(ys_ds, window_length=sec, polyorder=0)
		#y = savgol_filter(ys_matrix[:, 0], window_length=sec, polyorder=0)
		
		fig = plt.figure(figsize=(8, 8))
		gs  = fig.add_gridspec(3,3)
		ax1 = fig.add_subplot(gs[0, :])
		ax2 = fig.add_subplot(gs[1:, :2])
		ax3 = fig.add_subplot(gs[1:, 2])
		
		#ax1.plot(np.arange(0, y.size)/sec, ys_ds - y, color='#7570b3', linewidth=2)
		ax1.plot(np.arange(0, deg_ang.size)/sec, deg_ang, color='#7570b3', linewidth=2)
		ax1.set(xlabel="Time [m]", ylabel=r"Tail angle ($\circ$)")
		
		for i in range(xs_matrix.shape[0]):
			ax2.plot(xs_matrix[i, :], ys_matrix[i, :], color='#bbbbbb', alpha=0.2)
		im = ax2.scatter(xs_ds, ys_ds, c=cmap(norm(deg_ang)))
		#ax2.yaxis.set_inverted(True)
		ax2.set_axis_off() 
		divider = make_axes_locatable(ax2)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')
		
		ax3.hist(deg_ang, bins='fd', edgecolor='none', facecolor='#7570b3')
		ax3.set(xlabel=r"Tail angle ($\circ$)", ylabel="# count")
		
		fig.tight_layout()
		plt.suptitle(f"Fish {fish_id} ({genotype})")
		#pdb.set_trace()
		plt.savefig(os.path.join(dest_fld, f"fish-{fish_name}-tail-tracking-results.png"), dpi=300, format='png', bbox_inches="tight", transparent=False)
		plt.close('all')
		
		# plt.scatter(xs_ds, ys_ds - y, s=5)
		# plt.gca().invert_yaxis()
		# plt.show()
		# pdb.set_trace()
	
	df_mov = pd.DataFrame(df_mov, columns=['Fish',  'Genotype',  'Movements'])
	
	return df_mov


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
		
	df = plot_trajectory(files, fps, dest_fld)
	
	pdb.set_trace()
	sys.exit(0)
