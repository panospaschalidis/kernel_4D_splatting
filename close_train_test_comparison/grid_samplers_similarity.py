
import torch
import pdb
import torch.nn as nn 
from weighted_sampling_faster import kernel_based_sampler as kernel_F
from weighted_sampling import kernel_based_sampler as kernel
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
               out= kernel(
                   self.grid,
                   self.coords, 
                   self.interpolation_mode, 
                   self.padding_mode, 
                   self.align_corners
               ).squeeze().T
           print(out.sum())
           return out#self.linear(out)

    data = torch.load('../data.pth', weights_only=True)
    grid = data['grid']
    coords = data['coords']
    #grid = torch.rand(1, 5, 4, 4)
    #coords = torch.rand(1, 1, 200, 2)
    device = 'cuda'
    if 'gr_truth.pth' in os.listdir(os.getcwd()):
        GT = torch.load('gr_truth.pth')['gt']
    else:
        GT = torch.rand(len(data['coords'].squeeze()),1).to(device)
        #GT = torch.rand(200,1).to(device)
        torch.save({'gt':GT}, 'gr_truth.pth')
    l2_loss = nn.MSELoss()
    model = Network(16, 1, coords, grid)
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
        loss.backward(retain_graph=True)
        optimizer.step()
        if args.flag:
            print(f"model with pytorch grid_sampler---iter:{iteration}, loss:{loss}")
            print(f"weight_sum: {model.linear.weight.sum()}")
            print(f"weight_grad_sum: {model.linear.weight.grad.sum()}")
            print(f"bias_sum: {model.linear.bias.sum()}")
            print(f"bias_grad_sum: {model.linear.bias.grad.sum()}")
            print(f"coords_sum: {model.coords.sum()}")
            print(f"coords_grad_sum: {model.coords.grad.sum()}")
            #print(f"{model.coords.grad}")
            print(f"grid_sum: {model.grid.sum()}")
            print(f"grid_sum_grad: {model.grid.grad.sum()}")
            print(f"{model.grid.grad}")
        else:
            print(f"model with custom grid_sampler---iter:{iteration}, loss:{loss}")
            print(f"weight_sum: {model.linear.weight.sum()}")
            print(f"weight_grad_sum: {model.linear.weight.grad.sum()}")
            print(f"bias_sum: {model.linear.bias.sum()}")
            print(f"bias_grad_sum: {model.linear.bias.grad.sum()}")
            print(f"coords_sum: {model.coords.sum()}")
            print(f"coords_grad_sum: {model.coords.grad.sum()}")
            #print(f"{model.coords.grad}")
            print(f"grid_sum: {model.grid.sum()}")
            print(f"grid_sum_grad: {model.grid.grad.sum()}")
            print(f"{model.grid.grad}")
        #pdb.set_trace()
        optimizer.zero_grad()
    #for iteration in range(10):
    #    pred = model(X)
    #    loss = l2_loss(pred, GT)
    #    loss.backward()
    #    optimizer.step()
    #    if args.flag:
    #        print(f"model with pytorch grid_sampler---iter:{iteration}, loss:{loss}")
    #        print(f"weight_sum: {model.linear.weight.sum()}")
    #        print(f"weight_grad_sum: {model.linear.weight.grad.sum()}")
    #        print(f"bias_sum: {model.linear.bias.sum()}")
    #        print(f"bias_grad_sum: {model.linear.bias.grad.sum()}")
    #    else:
    #        print(f"model with custom grid_sampler---iter:{iteration}, loss:{loss}")
    #        print(f"weight_sum: {model.linear.weight.sum()}")
    #        print(f"weight_grad_sum: {model.linear.weight.grad.sum()}")
    #        print(f"bias_sum: {model.linear.bias.sum()}")
    #        print(f"bias_grad_sum: {model.linear.bias.grad.sum()}")
    #    optimizer.zero_grad()

