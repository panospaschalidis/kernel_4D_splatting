import torch
import os
import glob
import argparse
import visdom
from PIL import Image
from torchvision.transforms import ToTensor

if __name__=='__main__':
    parser = argparse.ArgumentParser('Visdom Renderer')
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None, 
        help='path wehre renderigns are being stored')
    args = parser.parse_args()
    transform = ToTensor()
    im0 = transform(Image.open(os.path.join(args.model_path,'train/ours_14000/gt/00094.png')))
    im1 = transform(Image.open(os.path.join(args.model_path,'train/ours_14000/renders/00094.png')))
    im3 = transform(Image.open(os.path.join(args.model_path,'test/ours_14000/gt/00094.png')))
    im4 = transform(Image.open(os.path.join(args.model_path,'test/ours_14000/renders/00094.png')))


    vis = visdom.Visdom()
    vis.image(torch.cat([torch.cat([im0, im1], dim=2), torch.cat([im3, im4], dim=2)], dim=1))


