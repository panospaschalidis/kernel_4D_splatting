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


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--env_double_splat", action="store_true")
    parser.add_argument("--env_gaus_hex", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--frame", type=str)
    parser.add_argument("--stage", type=str, default='coarse')
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    print("Before procceeing uncomment lines 74-74 from __init__ function of Load_hyper_data class from hyper_loader")
    if args.env_double_splat:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    
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
    pc = GaussianModel(dataset.sh_degree, hyper)
    pdb.set_trace()
    scene = Scene(dataset, pc, load_iteration=args.iteration, shuffle=False)
    print(f"Gaussian Rasterization from {len(pc._xyz)} Gaussians")
    cam_type=scene.dataset_type
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_viewpoints = scene.getTrainCameras()
    test_viewpoints = scene.getTestCameras()
    train_frames = [view.frame_name for view in train_viewpoints]
    test_frames = [view.frame_name for view in test_viewpoints]
    if args.train:
        view_index = train_frames.index(args.frame+'.png')
    if args.test:
        view_index = test_frames.index(args.frame+'.png')
    viewpoint_camera = train_viewpoints[view_index]
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    #pdb.set_trace()
    # Set up rasterization configuration
    scaling_modifier = 1.0    
    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    if args.env_gaus_hex:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color, dtype=torch.float32, device='cuda'),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=False
            )
    if args.env_double_splat:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor(bg_color, dtype=torch.float32, device='cuda'),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            gaussiansperpixel=False
            )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in args.stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in args.stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError


    #pdb.set_trace()
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    


    colors_precomp = None
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    if args.env_gaus_hex:
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final.cuda(),
            means2D = means2D.cuda(),
            shs = shs_final.cuda(),
            colors_precomp = colors_precomp,
            opacities = opacity.cuda(),
            scales = scales_final.cuda(),
            rotations = rotations_final.cuda(),
            cov3D_precomp = cov3D_precomp)
        Image.fromarray((np.array(rendered_image.permute(1,2,0).detach().cpu())*255).astype(np.uint8)).save('train_'+args.frame)
    if args.env_double_splat:
        im, radius, depth, conic, points_xy, list_0 = rasterizer( #diff_gaussian_rasterization #gaussian_per_pixel=False
        #im, radius, depth, conic, points_xy, list_0 = rasterizer( #diff_gaussian_rasterization #gaussian_per_pixel=True
        #im, radius, depth, gaus_color, list_0 = rasterizer( #diff_gaussian_rasterization_2
            means3D = means3D_final.cuda(),
            means2D = means2D.cuda(),
            shs = shs_final.cuda(),
            colors_precomp = colors_precomp,
            opacities = opacity.cuda(),
            scales = scales_final.cuda(),
            rotations = rotations_final.cuda(),
            cov3D_precomp = cov3D_precomp)
    pdb.set_trace()    
   # torch.save( {'means3D': means3D_final.detach().cpu(),
   # 'means2D': means2D.detach().cpu(),
   # 'shs': shs_final.detach().cpu(),
   # 'colors_precomp': colors_precomp,
   # 'opacities': opacity.detach().cpu(),
   # 'scales': scales_final.detach().cpu(),
   # 'rotations': rotations_final.detach().cpu(),
   # 'cov3D_precomp': cov3D_precomp}, 'gaussians_parameters_'+args.stage+'_'+args.frame+'.pth')
   # torch.save({'im':im.detach().cpu(), 'radius':radius.detach().cpu(), 'depth':depth.detach().cpu(), 'conic':conic.detach().cpu(), 'points_xy':points_xy.detach().cpu(), 'contrib'    : list_0}, 'save_for_later_'+args.stage+'_'+args.frame+'.pth')
    #pdb.set_trace()

