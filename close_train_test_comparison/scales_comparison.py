import sys
sys.path.append('/home/panos/workspace/internship_repos/kernel_4D_splatting')
import pdb
import numpy as np
import torch
from scene import Scene
import os
import math
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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
from utils.general_utils import strip_symmetric, build_scaling_rotation


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

parser = ArgumentParser(description="Training script parameters")
hp = ModelHiddenParams(parser)
lp = ModelParams(parser)
gaussian_overfit = GaussianModel(lp.sh_degree, hp)
gaussian_vanilla = GaussianModel(lp.sh_degree, hp)
gaussian_custom_kernel = GaussianModel(lp.sh_degree, hp)
gaussian_torch_kernel = GaussianModel(lp.sh_degree, hp)
path_overfit = os.path.join(os.path.dirname(os.getcwd()), 'output/custom/cretan_overfit_000168')
path_vanilla = os.path.join(os.path.dirname(os.getcwd()), 'output/custom/cretan')
path_custom_kernel = '/home/panos/workspace/internship_repos/4d-gaussian-hexplane/output/custom/cretan_coords_extra_grad_custom'
path_torch_kernel = '/home/panos/workspace/internship_repos/4d-gaussian-hexplane/output/custom/cretan_coords_extra_grad_torch'
pcl_path = 'point_cloud/iteration_14000/point_cloud.ply'
gaussian_overfit.load_ply(os.path.join(path_overfit, pcl_path))
gaussian_vanilla.load_ply(os.path.join(path_vanilla, pcl_path))
gaussian_custom_kernel.load_ply(os.path.join(path_custom_kernel, pcl_path))
gaussian_torch_kernel.load_ply(os.path.join(path_torch_kernel, pcl_path))

cov_overfit = build_covariance_from_scaling_rotation(
    gaussian_overfit.get_scaling,
    1.0,
    gaussian_overfit.get_rotation
)
cov_vanilla = build_covariance_from_scaling_rotation(
    gaussian_vanilla.get_scaling,
    1.0,
    gaussian_vanilla.get_rotation
)
cov_custom_kernel = build_covariance_from_scaling_rotation(
    gaussian_custom_kernel.get_scaling,
    1.0,
    gaussian_custom_kernel.get_rotation
)
cov_torch_kernel = build_covariance_from_scaling_rotation(
    gaussian_torch_kernel.get_scaling,
    1.0,
    gaussian_torch_kernel.get_rotation
)
