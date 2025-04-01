import sys
sys.path.append('/home/panos/workspace/internship_repos/kernel_4D_splatting')
import pdb
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
# import torch.multiprocessing as mp
import threading
import concurrent.futures
from PIL import Image

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    print("Before procceeing uncomment lines 74-74 from __init__ function of Load_hyper_data class from hyper_loader")
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    hyper = hyperparam.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cam_type=scene.dataset_type
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras()
    test_viewpoints = scene.getTestCameras()
    train_frames = [view.frame_name for view in train_viewpoints]
    test_frames = [view.frame_name for view in test_viewpoints]
    pdb.set_trace()
    train_view_index = train_frames.index('frame_000168.png')
    test_view_index = test_frames.index('frame_000169.png')
    train_view = train_viewpoints[train_view_index]
    train_rendering = render(train_view, gaussians, pipe, background,stage='coarse',cam_type=cam_type)["render"].detach().permute(1,2,0).cpu()
    test_view = test_viewpoints[test_view_index]
    test_rendering = render(test_view, gaussians, pipe, background,stage='coarse', cam_type=cam_type)["render"].detach().permute(1,2,0).cpu()
    Image.fromarray((np.array(train_rendering)*255).astype(np.uint8)).save('train_000168.png')
    Image.fromarray((np.array(test_rendering)*255).astype(np.uint8)).save('test_000169.png')

