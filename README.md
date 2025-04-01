## kernel_4D_splatting

This repo is built on top of
[4D_Gaussians](https://github.com/hustvl/4DGaussians) and provides an
alternative for better generalization during inference about novel view
synthesis from bounded monocular videos.  A conditioned to time hybrid
deformation network comprised of a
[HexPlane](https://github.com/Caoang327/HexPlane) representation and swallow
MLPs predicts the position rotation and scaling deformations of a set of
canonical Gaussians. The deormed Gaussians are then splatted and optimization
proceeds.

The term **kernel** in our framework comes from the kernel function that we
utilize during feature interpolation.  Specifically following
[GTK](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Grounding_and_Enhancing_Grid-based_Models_for_Neural_Fields_CVPR_2024_paper.pdf)
we avoid HexPlane features bilinear interpoaltion as was performed in
`4DGaussians`.  Instead a [Multiplicative Filter
Network](https://openreview.net/forum?id=OmtmcPkkhT) is employed as a
**kernel/similarity** function of the combined features.

## Pipeline

## Installation
```
conda create --name ellipsis python=3.10
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```
Regarding submodules directory we have made the following additions
1. **diff_gaussian_rasterization_mip**
    Instead of the standard [GS](https://github.com/graphdeco-inria/gaussian-splatting) pipeline we use
    [mip splatting](https://arxiv.org/abs/2311.16493) that we have implemented on top of the vanilla one
    in submodules/diff_gaussian_rasterization.
2. **weighted_sampling_mfn_CUDA**
    We provide a grid sampling module that performs non-linear interpolation from the processed grids 
    during the forward pass and backpropagates the gradients to both the grids and Gaussian coordinates.
3. **mfn_CUDA_softmax**
    The aforementioned kenrel function that provides the non-linear interpolation is a 2-layer **MFN** network.
    To accelerate training performance we have created custom CUDA kernels for estimating MFN's  `output` as well as 
    loss gradients with respect to network parameters. Moreover, we provided the `Jacobian` as well since it is required
    during Gaussians cords gradient estimation in **2**.
4. **grid_indexing**
    mfn network is activated not only with the Gaussian coords but the row and column indices that define the location where they are projected. 
    That being said this is a helper module written in C++ and OpenMP to further accelerate this indexing procedure.
```
cd submodules

```


## Data 
From the available monocular videos  [CVD](https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/), [DAVIS](https://davischallenge.org/), [NVIDIA](https://gorokee.github.io/jsyoon/dynamic_synth/), [Google](https://augmentedperception.github.io/deepviewvideo/) we selected `DAVIS` and `NVIDIA` as these are the only ones
which correspond to actual dynamic motion fields. We tested our method and baselines to even more challenging 
custom scenes as seen below in [Training](## Training) and [Evaluation](## Evaluation).
To model your own scenes or any scene from the above datasets do the following.
Run COLMAP
Having obtained the colmap parameters run custom_colmap.sh to form a data directory in the same style as in [NeRFies](https://github.com/google/nerfies).

## Training

![](./media/random_gaussians.png)

## Evaluation

