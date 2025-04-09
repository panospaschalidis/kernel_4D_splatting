"""standard libraries"""
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import os
import glob
import lpips
import argparse
import pdb
import time
import psutil
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description = "comparing images"
)
parser.add_argument(
    "--target",
    type=str,
    help="target directory"
)
parser.add_argument(
    "--predicted",
    type=str,
    help="predicted directory"
)
args = parser.parse_args()
target_directory = args.target
pred_directory = args.predicted
target_paths = sorted(glob.glob(os.path.join(target_directory, '*')))
pred_paths =  sorted(glob.glob(os.path.join(pred_directory, '*')))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net='vgg').to(device)
from torchmetrics import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
from torchmetrics import PeakSignalNoiseRatio
psnr = PeakSignalNoiseRatio().to(device)
# define indices of target images
transform = ToTensor()
all_psnr = []
all_ssim = []
all_lpips = []
#print('before memory % used:', psutil.virtual_memory()[2])
tbar = tqdm(range(len(target_paths)))
for val, (target_path, pred_path) in enumerate(zip(target_paths, pred_paths)):
    #print(val)
    #target_im = torch.tensor(
    #    np.load(target_path)['rgb'][ind,...,:3]
    #                        ).div(255).permute(0,3,1,2)
    ##print('target loaded memory % used:', psutil.virtual_memory()[2])
    #pred_im = torch.tensor(np.load(pred_path)['rgb']).permute(0,3,1,2)
    #print('prediction loaded memory % used:', psutil.virtual_memory()[2])
    target_im = transform(Image.open(target_path)).unsqueeze(0).to(device)
    pred_im = transform(Image.open(pred_path)).unsqueeze(0).to(device)
    all_psnr.append(psnr(pred_im, target_im).unsqueeze(-1))
    #print('psnr memory % used:', psutil.virtual_memory()[2])
    all_ssim.append(ssim(pred_im, target_im).unsqueeze(-1))
    #print('ssim memory % used:', psutil.virtual_memory()[2])
    all_lpips.append(lpips(pred_im, target_im).detach().unsqueeze(-1))
    #print('lpips memory % used:', psutil.virtual_memory()[2])
    tbar.update(1)

print(f"for {os.path.relpath(pred_directory)}")
print(f"psnr: {torch.sum(torch.cat(all_psnr,dim=-1))/len(all_psnr)}")
print(f"ssim: {torch.sum(torch.cat(all_ssim,dim=-1))/len(all_ssim)}")
print(f"lpips: {torch.sum(torch.cat(all_lpips, dim=-1))/len(all_lpips)}")
