#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import pdb
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        ## uncomment in case of double_splat env in order to obtain conic
        #raster_settings = GaussianRasterizationSettings(
        #    image_height=int(viewpoint_camera.image_height),
        #    image_width=int(viewpoint_camera.image_width),
        #    tanfovx=tanfovx,
        #    tanfovy=tanfovy,
        #    bg=bg_color,
        #    scale_modifier=scaling_modifier,
        #    viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        #    projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        #    sh_degree=pc.active_sh_degree,
        #    campos=viewpoint_camera.camera_center.cuda(),
        #    prefiltered=False,
        #    gaussiansperpixel=False
        #)
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

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
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError

    if 'filter_3D' not in dir(pc):
        scales_final = pc.scaling_activation(scales_final)
        opacity = pc.opacity_activation(opacity_final)
    else:
        opacity = pc.get_opacity_with_3D_filter(opacity_final, scales_final)
        scales_final = pc.get_scaling_with_3D_filter(scales_final)
    
    rotations_final = pc.rotation_activation(rotations_final)
    #R = build_rotation(rotations_final[52502,:].unsqueeze(0)).squeeze()
    #S = torch.eye(3, device='cuda') * scales_final[52502,:]
    #M = S@R
    #Sigma = M.T@M
    #limx = 1.3/tanfovx
    #limy = 1.3/tanfovy
    #viewmatrix =viewpoint_camera.world_view_transform.cuda()
    #m3D_homog = torch.cat([means3D_final[52502,:],torch.ones(1, device='cuda')])
    #t = (m3D_homog.unsqueeze(0) @ viewmatrix).squeeze()
    #txtz = t[0]/t[2]
    #tytz = t[1]/t[2]
    #t_x = min(limx, max(-limx, txtz))*t[2]
    #t_y = min(limy, max(-limy, tytz))*t[2]
    #image_height=int(viewpoint_camera.image_height)
    #image_width=int(viewpoint_camera.image_width)
    #focal_y = image_height / (2 * tanfovy)
    #focal_x = image_width / (2 * tanfovx)
    #J = torch.tensor([
    #    [focal_x/t[2], 0, -(focal_x*t[0])/(t[2]**2)],
    #    [0, focal_y/t[2], -(focal_y*t[1])/(t[2]**2)],
    #    [0,0,0]
    #], device='cuda')
    #W = viewmatrix.T[:3,:3]
    #T = W@J
    #cov = T.T 
    #flag  = (torch.eye(3, device='cuda')==0)
    #Vrk = (Sigma.triu() * flag).T + Sigma.triu()
    #cov = T.T @ Vrk.T @ T
    #cov[0][0] += 0.3
    #cov[1][1] += 0.3
    #cov2D = torch.tensor([cov[0][0]+0.3, cov[0][1], cov[1][1]+0.3])
    #det = cov2D[0]*cov2D[2] - cov2D[1]**2
    #conic = torch.tensor([cov2D[2]/det, -cov2D[1]/det, cov2D[0]/det])
    #print(Sigma)
    #print(S)
    #pdb.set_trace()
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    #rendered_image, radii, depth = rasterizer(
    #    means3D = means3D_final[:121670,:],
    #    means2D = means2D[:121670,:],
    #    shs = shs_final[:121670,:],
    #    colors_precomp = colors_precomp,
    #    opacities = opacity[:121670,:],
    #    scales = scales_final[:121670,:],
    #    rotations = rotations_final[:121670,:],
    #    cov3D_precomp = cov3D_precomp)
    #rendered_image, radii, depth = rasterizer(
    #    means3D = means3D_final[5216:,:],
    #    means2D = means2D[5216:,:],
    #    shs = shs_final[5216:,:],
    #    colors_precomp = colors_precomp,
    #    opacities = opacity[5216:,:],
    #    scales = scales_final[5216:,:],
    #    rotations = rotations_final[5216:,:],
    #    cov3D_precomp = cov3D_precomp)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}
    
    ## uncomment in case of double_splat env in order to obtain conic
    #rendered_image, radii, depth, conic, _, _ = rasterizer(
    #    means3D = means3D_final,
    #    means2D = means2D,
    #    shs = shs_final,
    #    colors_precomp = colors_precomp,
    #    opacities = opacity,
    #    scales = scales_final,
    #    rotations = rotations_final,
    #    cov3D_precomp = cov3D_precomp)
    #
    #return {"render": rendered_image,
    #        "viewspace_points": screenspace_points,
    #        "visibility_filter" : radii > 0,
    #        "radii": radii,
    #        "depth":depth,
    #        "conic":conic}

