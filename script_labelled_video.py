"""
Script for analyzing lighting-pose predictions.
"""

import os 

import numpy as np 

from lightning_pose.train import train
from lightning_pose.model import Model 
import pandas as pd 

from lightning_pose.utils.predictions import generate_labeled_video 

from omegaconf import OmegaConf

import matplotlib.pyplot as plt

name_video = '2022-03-24-09-04-01'

main_folder = "/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48"

video_file = "/home/enrique/lp-one-photon/videos/" + name_video + ".mp4"


cfg = OmegaConf.load(os.path.join(main_folder, "config.yaml"))

model = Model.from_dir(main_folder)

model.predict_on_video_file(video_file, generate_labeled_video=False)


df = pd.read_csv(os.path.join(main_folder, "video_preds/" + name_video + ".csv"), header=[1, 2], index_col=0)


labeled_mp4_file = "/home/enrique/lp-one-photon/outputs/2025-04-23/12:16:48/video_preds/labeled_videos/" + name_video + "_labeled.mp4"

generate_labeled_video(
	video_file = video_file,
	preds_df = df,
	output_mp4_file=labeled_mp4_file, 
	confidence_thresh_for_vid=cfg.eval.confidence_thresh_for_vid,
	colormap=cfg.eval.get("colormap", "cool"),
	)
