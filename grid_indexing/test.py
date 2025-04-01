nimport os
os.environ["OMP_NUM_THREADS"] = "3"  # Example for setting 4 threads
import argparse
import pdb
from tqdm import tqdm
import torch
import aggregator
parser = argparse.ArgumentParser("pytorch parallelism on the test")
parser.add_argument('--gr','-grain_size', type=int, default = 0 )

args = parser.parse_args()
data = torch.load("/home/panagiotis/internship/meetings/27th_meeting/data.pth")

# when using C++
R_index = aggregator.mfn_forward(data['coords'].squeeze(), 64, 64, args.gr)
#index = aggregator.mfn_forward(data['coords'].squeeze())

W = torch.tensor(64, dtype=torch.float32)
H = torch.tensor(64, dtype=torch.float32)
# when using pure python for speed comparison
#coords = data['coords'].squeeze().clone()
for i in range(len(coords)):
    x = coords[i,0]
    y = coords[i,1]
    norm_x = ((x+1)/2)*(W-1)
    norm_y = ((y+1)/2)*(H-1)
    cnorm_x = min(W-1, max(norm_x,0))
    cnorm_y = min(H-1, max(norm_y,0))
    coords[i,0] = cnorm_x
    coords[i,1] = cnorm_y
    if i==29315:
        pdb.set_trace()
new_coords = coords.floor().clone()
R_index = torch.cat([((new_coords[:,0]) + (new_coords[:,1])*W).unsqueeze(-1), ((new_coords[:,0]+1) + (new_coords[:,1])*W).unsqueeze(-1),((new_coords[:,0]) + (new_coords[:,1]+1)*W).unsqueeze(-1), ((new_coords[:,0]+1) + (new_coords[:,1]+1)*W).unsqueeze(-1)], dim=-1)


coords = data['coords'].squeeze().clone()
grid = data['grid'].squeeze().clone()
interp = data['interp'].squeeze().clone()
Rinterp = torch.empty(16,38890)
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
    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    #print(f"ix_nw: {ix_nw}")
    #print(f"iy_nw: {iy_nw}")
    #print(f"nw: {nw}")
    #print(f"ix_ne: {ix_ne}")
    #print(f"iy_ne: {iy_ne}")
    #print(f"ne: {ne}")
    #print(f"ix_sw: {ix_sw}")
    #print(f"iy_sw: {iy_sw}")
    #print(f"sw: {sw}")
    #print(f"ix_se: {ix_se}")
    #print(f"iy_se: {iy_se}")
    #print(f"se: {se}")
    Rinterp[:,i] = grid[:,int(iy_nw), int(ix_nw)]*nw + grid[:,int(iy_ne), int \
    (ix_ne)]*ne + grid[:,int(iy_sw), int(ix_sw)]*sw + grid[:,int(iy_se), int(ix_se)]*se


coords = data['coords'].squeeze().clone()
grid = data['grid'].squeeze().clone()
interp = data['interp'].squeeze().clone()
Rinterp = torch.empty(16,38890)
for i in tqdm(range(len(coords))):
    x = coords[i,0]
    y = coords[i,1]
    norm_x = ((x+1)/2)*(W-1)
    norm_y = ((y+1)/2)*(H-1)
    ix = min(W-1, max(norm_x,torch.tensor(0).to(torch.float32)
    ))
    iy = min(H-1, max(norm_y,torch.tensor(0).to(torch.float32)
    ))
    #print(f"ix: {ix}")
    #print(f"iy: {iy}")
    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw +1 
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw +1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1
    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    print(f"ix_nw: {ix_nw}")
    print(f"iy_nw: {iy_nw}")
    print(f"nw: {nw}")
    print(f"ix_ne: {ix_ne}")
    print(f"iy_ne: {iy_ne}")
    print(f"ne: {ne}")
    print(f"ix_sw: {ix_sw}")
    print(f"iy_sw: {iy_sw}")
    print(f"sw: {sw}")
    print(f"ix_se: {ix_se}")
    print(f"iy_se: {iy_se}")
    print(f"se: {se}")
    Rinterp[:,i] = grid[:,int(iy_nw), int(ix_nw)]*nw + grid[:,int(iy_ne), int \
    (ix_ne)]*ne + grid[:,int(iy_sw), int(ix_sw)]*sw + grid[:,int(iy_se), int(ix_se)]*se

