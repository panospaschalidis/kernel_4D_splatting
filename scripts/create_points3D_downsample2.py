import os 
import open3d as o3d
import shutil
import numpy as np


os.makedirs("colmap", exist_ok=True)

os.makedirs("colmap/dense/workspace", exist_ok=True)
shutil.copy(
    "colmap_dense/dense/0/fused.ply",
    "colmap/dense/workspace"
    )
pcd = o3d.io.read_point_cloud("colmap/dense/workspace/fused.ply")

points = np.array(pcd.points)

colors = np.array(pcd.colors)

normals = np.array(pcd.normals)

flag = np.sum(colors, axis=1) >0

points = np.array(pcd.points)[flag]

colors = np.array(pcd.colors)[flag]

normals = np.array(pcd.normals)[flag]

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)

pcd.colors = o3d.utility.Vector3dVector(colors)

pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.io.write_point_cloud("points3D_downsample2.ply", pcd)


