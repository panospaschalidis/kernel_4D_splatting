from tqdm import tqdm
import torch
import pdb
import torch.nn as nn 
from weighted_sampling_faster import kernel_based_sampler as kernel_F
from weighted_sampling import kernel_based_sampler as kernel
import grid_indexing
import torch.nn.functional as F
import time
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing grid samplers similarity")
    parser.add_argument("--flag", action="store_true"),
    parser.add_argument("--iter",type=int , default=1)
    args = parser.parse_args()
    class Network(nn.Module):
        def __init__(self, inpt, out, coords, grid):
            super().__init__()
            self.linear= nn.Linear(inpt,out)
            self.coords = torch.nn.Parameter(coords)
            self.grid = torch.nn.Parameter(grid)
            self.interpolation_mode = 'bilinear'
            self.padding_mode = 'border'
            self.align_corners=True
        
        def forward(self, flag):
           if flag:
               out = F.grid_sample(
                   self.grid, 
                   self.coords, 
                   align_corners=self.align_corners, 
                   mode=self.interpolation_mode, 
                   padding_mode=self.padding_mode
               ).squeeze().T
           else:
               out= kernel_F(
                   self.grid,
                   self.coords, 
                   self.interpolation_mode, 
                   self.padding_mode, 
                   self.align_corners
               ).squeeze().T
           print(out.sum())
           return out#self.linear(out)

   # data = torch.load('../data.pth', weights_only=True)
   # grid = data['grid']
   # coords = data['coords']
    grid = torch.rand(1, 5, 4, 4)
    coords = torch.rand(1, 1, 200, 2)
    device = 'cuda'
    if 'gr_truth.pth' in os.listdir(os.getcwd()):
        GT = torch.load('gr_truth.pth')['gt']
    else:
        #GT = torch.rand(len(data['coords'].squeeze()),1).to(device)
        GT = torch.rand(200,1).to(device)
        torch.save({'gt':GT}, 'gr_truth.pth')
    l2_loss = nn.MSELoss()
    model = Network(5, 1, coords, grid)
    #if 'input.pth' in os.listdir(os.getcwd()):
    #    X = torch.load('input.pth')['input']
    #    GT = torch.load('input.pth')['gt']
    #else:
    #    X = torch.rand(100, 16, device=device)
    #    GT = torch.rand(100, 1, device=device)
    #    torch.save({'input':X, 'gt':GT}, 'input.pth')
    if 'model.pth' in os.listdir(os.getcwd()):
        model.load_state_dict(torch.load('model.pth', weights_only=True))
    else:
        torch.save(model.state_dict(), 'model.pth')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, eps=1e-15)
    #print(f"initial_coords_sum: {coords.sum()}")
    #print(f"initial_grid_sum: {grid.sum()}")
    for iteration in range(args.iter):
        pred = model(args.flag)
        loss = l2_loss(model.linear(pred), GT)

    
    #grad_pred = torch.autograd.grad(loss, pred, retain_graph=True)[0]
    grad_pred = torch.load('grad.pth').squeeze().permute(1,0)
    R_index = grid_indexing.indexing(model.coords.squeeze().detach().cpu(), 4, 4, 4)
    W = torch.tensor(4, dtype=torch.float32)
    H = torch.tensor(4, dtype=torch.float32)
    grad_grid = torch.zeros_like(grid, device=device)
    coords = model.coords.squeeze()
    list_ = []
    for i in tqdm(range(len(coords))):
        x = coords[i,0]
        y = coords[i,1]
        norm_x = ((x+1)/2)*(W-1)
        norm_y = ((y+1)/2)*(H-1)
        ix = min(W-1, max(norm_x,0))
        iy = min(H-1, max(norm_y,0))
        #print(f"ix: {ix}")
        #print(f"iy: {iy}")
        iy_nw = (R_index[i,0]//W)
        ix_nw = R_index[i,0] - (iy_nw)*W
        iy_ne = (R_index[i,1]//W)
        ix_ne = R_index[i,1] - (iy_ne)*W
        iy_sw = (R_index[i,2]//W)
        ix_sw = R_index[i,2] - (iy_sw)*W
        iy_se = (R_index[i,3]//W)
        ix_se = R_index[i,3] - (iy_se)*W
        print(f"{ix}, {iy}, {ix_nw}, {iy_nw}, {ix_ne}, {iy_ne}, {ix_sw}, {iy_sw}, {ix_se}, {iy_se}")
        nw = (ix_se - ix)    * (iy_se - iy) if (ix_nw<64 and iy_nw<64) else 0
        ne = (ix    - ix_sw) * (iy_sw - iy) if (ix_ne<64 and iy_ne<64) else 0
        sw = (ix_ne - ix)    * (iy    - iy_ne) if (ix_sw<64 and iy_sw<64) else 0
        se = (ix    - ix_nw) * (iy    - iy_nw) if (ix_se<64 and iy_se<64) else 0
        if nw!=0:
            grad_grid[...,int(iy_nw), int(ix_nw)] += nw*grad_pred[i,...]
        if ne!=0:
            grad_grid[...,int(iy_ne), int(ix_ne)] += ne*grad_pred[i,...]
        if sw!=0:
            grad_grid[...,int(iy_sw), int(ix_sw)] += sw*grad_pred[i,...]
        if se!=0:
            grad_grid[...,int(iy_se), int(ix_se)] += se*grad_pred[i,...]
        if iy_nw==1 and ix_nw==2:
            list_.append(nw*grad_pred[i,...])
        if iy_ne==1 and ix_ne==2:
            list_.append(ne*grad_pred[i,...])
        if iy_sw==1 and ix_sw==2:
            list_.append(sw*grad_pred[i,...])
        if iy_se==1 and ix_se==2:
            list_.append(se*grad_pred[i,...])

    print(grad_grid)    

