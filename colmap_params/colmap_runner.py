# Inspired by consistent video depth estimation
# https://github.com/facebookresearch/consistent_depth

import argparse

import pdb
import argparse
import logging
import os
from os.path import join as pjoin
import subprocess
import sys

import numpy as np

def dense_depth_suffix():
    return '.geometric.bin'

def db_path(workspace):
    return pjoin(workspace, 'database.db')

def _sparse_dir(workspace, model_index=None):
    p = pjoin(workspace, 'sparse')
    if model_index is None:
        return p
    return pjoin(p, str(model_index))

def _dense_dir(workspace, model_index=None):
    p = pjoin(workspace, 'dense')
    if model_index is None:
        return p
    return pjoin(p, str(model_index))

def pose_init_dir(workspace):
    return pjoin(workspace, 'pose_init')

def check_sparse(sparse_model_dir: str):
    return any(
        all(
            (os.path.isfile(pjoin(sparse_model_dir, name))
            for name in ['cameras' + ext, 'images' + ext])
        )
        for ext in ['.bin', '.txt']
    )

def check_dense(dense_model_dir: str, image_path: str, valid_ratio=1):
    assert valid_ratio <= 1

    depth_fmt = pjoin(
        dense_model_dir, 'stereo', 'depth_maps', '{}' + dense_depth_suffix()
    )
    color_names = os.listdir(image_path)

    num_valid = np.sum(os.path.isfile(depth_fmt.format(n)) for n in color_names)
    return (num_valid / len(color_names)) >= valid_ratio


if __name__=='__main__':
    parser = argparse.ArgumentParser('Colmap_Runner')
    parser.add_argument('instance_path', help='image path')
    parser.add_argument(
        '--mask_path',
        help='path for mask to exclude feature extration from those regions',
        default=None,
    )
    parser.add_argument(
        '--dense_max_size', type=int, help='Max size for dense COLMAP', default=384,
    )
    parser.add_argument(
        '--sparse', help='disable dense reconstruction', action='store_true'
    )
    parser.add_argument(
        '--initialize_pose', help='Intialize Pose', action='store_true'
    )
    parser.add_argument(
        '--camera_params', help='prior camera parameters', default=None
    )
    parser.add_argument(
        '--camera_model', help='camera_model', default='SIMPLE_PINHOLE'
    )
    parser.add_argument(
        '--refine_intrinsics',
        help='refine camera parameters. Not used when camera_params is None',
        action='store_true'
    )
    parser.add_argument(
        '--matcher', choices=['exhaustive', 'sequential'], default='exhaustive',
        help="COLMAP matcher (''exhaustive' or 'sequential')"
    )
    
    args = parser.parse_args()
    workspace_path = os.path.join(args.instance_path, 'colmap_dense')
    image_path = os.path.join(args.instance_path, 'color_full') # earlier created by video2im_seq.py
    os.makedirs(workspace_path, exist_ok=True)
    # extract_features
    cmd = [
        'colmap',
        'feature_extractor',
        '--database_path', db_path(workspace_path),
        '--image_path', image_path,
        '--ImageReader.camera_model', args.camera_model,
        '--SiftExtraction.use_gpu', '1',
        '--SiftExtraction.gpu_index', '0',
    ]

    if args.camera_params:
        cmd.extend(['--ImageReader.camera_params', args.camera_params])

    if args.mask_path:
        cmd.extend(['--ImageReader.mask_path', args.mask_path])

    if args.initialize_pose:
        cmd.extend(['--SiftExtraction.num_threads', '1'])
        cmd.extend(['--SiftExtraction.gpu_index', '0'])


    subprocess.run(cmd)

    # matching
    cmd = [
        'colmap',
        f'{args.matcher}_matcher',
        '--database_path', db_path(workspace_path),
        '--SiftMatching.guided_matching', '1',
        '--SiftMatching.use_gpu', '1',
        '--SiftMatching.gpu_index', '0', 
    ]
    if args.matcher == 'sequential':
        cmd.extend([
            '--SequentialMatching.overlap', '50',
            '--SequentialMatching.quadratic_overlap', '0',
        ])
    subprocess.run(cmd)

    if args.initialize_pose:
        if not check_sparse(_sparse_dir(workspace_path, model_index=0)):
            pose_init_dir = pose_init_dir(workspace_path)
            assert check_sparse(pose_init_dir)

            sparse_dir = sparse_dir(workspace_path, model_index=0)
            os.makedirs(sparse_dir, exist_ok=True)
            cmd = [
                'colmap',
                'point_triangulator',
                '--database_path', db_path(workspace_path),
                '--image_path', image_path,
                '--output_path', sparse_dir,
                '--input_path', pose_init_dir,
                '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_local_max_num_iterations', '0',
                '--Mapper.ba_global_max_num_iterations', '1',
            ]
            subprocess.run(cmd)
    else:
        if not check_sparse(_sparse_dir(workspace_path, model_index=0)):

            sparse_dir = _sparse_dir(workspace_path)
            os.makedirs(sparse_dir, exist_ok=True)
            #--------------------------------------------------------------------#
            cmd = [
                'colmap',
                'mapper',
                '--database_path', db_path(workspace_path),
                '--image_path', image_path,
                '--output_path', sparse_dir,
                # add the following options for KITTI evaluation. Should help in general.
                '--Mapper.abs_pose_min_inlier_ratio', '0.5',
                '--Mapper.abs_pose_min_num_inliers', '50',
                '--Mapper.init_max_forward_motion', '1',
                '--Mapper.ba_local_num_images', '15',
            ]
            if args.camera_params and not args.refine_intrinsics:
                cmd.extend([
                    '--Mapper.ba_refine_focal_length', '0',
                    '--Mapper.ba_refine_extra_params', '0',
                ])
            subprocess.run(cmd)
    
    # MVS
    if not args.sparse:
        dense_dir = _dense_dir(workspace_path, model_index=0)
        if not check_dense(dense_dir, image_path):
            os.makedirs(dense_dir, exist_ok=True)
            cmd = [
                'colmap',
                'image_undistorter',
                '--image_path', image_path,
                '--input_path', _sparse_dir(workspace_path, model_index=0),
                '--output_path', dense_dir,
                '--output_type', 'COLMAP',
                '--max_image_size', str(args.dense_max_size),
            ]
            subprocess.run(cmd)

            cmd = [
                'colmap',
                'patch_match_stereo',
                '--workspace_path', dense_dir,
                '--workspace_format', 'COLMAP',
                '--PatchMatchStereo.max_image_size', str(args.dense_max_size),
            ]
            subprocess.run(cmd)

            #--------------------------------------------------------------------#
            cmd = [
                'colmap',
                'stereo_fusion',
                '--workspace_path', dense_dir,
                '--workspace_format', 'COLMAP',
                '--input_type',  'geometric',
                '--output_path', f'{dense_dir}/fused.ply'
            ]
            subprocess.run(cmd)
            #--------------------------------------------------------------------#
        
