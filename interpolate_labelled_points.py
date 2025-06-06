"""
Script that interpolates the labelled points of the tracking 
peformed by using Lightning Pose.
"""
import numpy as np
import pandas as pd 

import os
import glob
import ntpath
import sys 

import h5py
import json 

from typing import List
from datetime import date
import locale 

import copy

from typing import Tuple

import pdb

from IPython import get_ipython
import pickle

import re 
import pdb
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from functools import partial 

from scipy.interpolate import make_splprep

def update(frame, xs, ys, ln):
	# for each frame, update the data stored on each artist.
	x = xs[frame, :]
	y = ys[frame, :]
	spl, u = make_splprep([x, y], s=0.1)
	pdb.set_trace()
	# update the line plot:
	ln.set_data(spl(u)[0,:], spl(u)[1, :])
	return ln,

def interpolate_points(files:str):
	for file_data in files:
		df = pd.read_csv(file_data, header=[1, 2], index_col=0)
		keypoints_arr = np.reshape(df.to_numpy(), [df.shape[0], -1, 3])
		xs_arr = keypoints_arr[:, :, 0]
		ys_arr = keypoints_arr[:, :, 1]
		
		fig, ax = plt.subplots()
		spl, u = make_splprep([xs_arr[0,:], ys_arr[0,:]], s=0.1)
		line1, = ax.plot(spl(u)[0, :], spl(u)[1, :], 'r')
		# set "partial" version of the animate function that presets all
		# the keyword arguments to animate (apart from i)
		anifunc = partial(update, xs=xs_arr, ys=ys_arr, ln=line1)
		
		# make the animation
		ani = animation.FuncAnimation(fig,  # pass the figure object
										anifunc,  # pass the animation function
										frames=np.arange(1, 102700),
										interval=0.01,
										blit=True,
										)

		pdb.set_trace() 
	
	return 


if __name__ == '__main__':
	preds_folder = '/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48/video_preds/'
	
	files      = []
	for p, d, f in os.walk(preds_folder):
		for file in f:
			if file.endswith('.csv'):
				if file.endswith('_temporal_norm.csv'):
					pass
				else:
					files.append(os.path.join(p, file))
	
	interpolate_points(files)
	pdb.set_trace()
