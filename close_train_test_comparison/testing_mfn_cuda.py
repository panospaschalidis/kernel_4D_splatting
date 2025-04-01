import pdb
import torch
import torch.nn as nn
import grid_indexing
import time
import os

from torch.autograd.functional import jacobian


class Network(nn.Module):
    def __init__(self,inpt,hidden,out, pos):
        super().__init__()
        self.pos = pos
        self.linear_0 = nn.Linear(inpt*pos, hidden)
        self.non_linear_in_0 = nn.Linear(1,hidden)
        self.out_linear = nn.Linear(hidden, out)
    
    @staticmethod
    def pos_encoding(x,d):
          out = []
          out.append(x)
          for i in range(1,d+1):
              if i%2:
                  out.append(torch.sin(2**(i/2)*torch.pi*x))
              else:
                  out.append(torch.cos(2**(i/2)*torch.pi*x))
          return torch.cat(out, dim =1)


    def forward(self,x, indices):
        s = 0
        list_o = []
        pos_x = self.pos_encoding(x,self.pos)
        x = pos_x[:,2:]
        common = self.linear_0(x)
        for i in range(4):
            out = common *torch.sin(self.non_linear_in_0(indices[:,i].unsqueeze(-1)))
            o = self.out_linear(out)
            s +=o
            list_o.append(o)
        #phi = [o/s for o in list_o]
        #phi = [o for o in list_o]
        #return torch.cat(phi, dim=-1)
        phi = torch.cat([o for o in list_o], dim=-1)
        #return torch.softmax(phi, dim=1)
        phi = phi - phi.max(dim=1)[0].unsqueeze(-1)
        return nn.functional.softmax(phi, dim=1)

model = Network(2, 256, 1,10)
if 'model_mfn.pth' in os.listdir(os.getcwd()):
    model.load_state_dict(torch.load('model_mfn.pth', weights_only=True))
else:
    torch.save(model.state_dict(), 'model_mfn.pth')
model.to('cuda')
data = torch.load('../data.pth')
indices = grid_indexing.indexing(data['coords'].squeeze(), 64, 64, 4).to('cuda')[:100,:]
#--------------------BATCH_SIZE 1--------------------#
#x = data['coords'].squeeze().to('cuda')[0,:][None,:]
#phi =  model(x, indices.to(torch.float32)[0,:][None,:])
#start = time.time()
#phi =  model(x, indices.to(torch.float32)[0,:][None,:])
#print('Forward pass {:.3f}'.format(time.time()-start))
#--------------------BATCH_SIZE 1--------------------#
#--------------------BATCH_SIZE>1--------------------#
x = data['coords'].squeeze().to('cuda')[:100,:]
#x = model.pos_encoding(x, model.pos)[:,2:]
start = time.time()
phi =  model(x, indices.to(torch.float32))
torch.cuda.synchronize()
print('Forward pass {:.3f}'.format(time.time()-start))
#--------------------BATCH_SIZE>1--------------------#
if 'gt_mfn.pth' in os.listdir(os.getcwd()):
    gt_data = torch.load('gt_mfn.pth')
    GT = gt_data['GT']
    w1 = gt_data['w1'] 
    w2 = gt_data['w2'] 
    w3 = gt_data['w3'] 
    w4 = gt_data['w4'] 
else:
    gt_data = {}
    GT = torch.rand((x.shape[0],indices.shape[1]), device='cuda')
    gt_data['GT'] = GT
    #w1 = torch.rand(1,4, device='cuda')
    #w2 = torch.rand(1,4, device = 'cuda')
    #w3 = torch.rand(1,4, device = 'cuda')
    #w4 = torch.rand(1,4, device='cuda')
    w1 = torch.rand(x.shape[0],4, device='cuda')
    w2 = torch.rand(x.shape[0],4, device = 'cuda')
    w3 = torch.rand(x.shape[0],4, device = 'cuda')
    w4 = torch.rand(x.shape[0],4, device='cuda')
    gt_data['w1'] = w1
    gt_data['w2'] = w2
    gt_data['w3'] = w3
    gt_data['w4'] = w4
    torch.save(gt_data, 'gt_mfn.pth')
#--------------------BATCH_SIZE 1--------------------#
#out = phi[0]*w1 +phi[1]*w2 + phi[2]*w3 + phi[3]*w4
#--------------------BATCH_SIZE>1--------------------#
out = phi[0][0]*w1 +phi[0][1]*w2 + phi[0][2]*w3 + phi[0][3]*w4
l2_loss = nn.MSELoss()
loss = l2_loss(out, GT)
start = time.time()
loss.backward(retain_graph=True)
torch.cuda.synchronize()
print('Backward pass {:.3f}'.format(time.time()-start))
#out 
#print("-------------out_grad---------------")
#print(torch.autograd.grad(loss, out, retain_graph = True))
#print((2/(out-GT).numel())*(out-GT))
grad_out = (2/(out-GT).numel())*(out-GT)


import mfn_softmax_CUDA
x = data['coords'].squeeze().to('cuda')[:100,:]
x = model.pos_encoding(x, model.pos)[:,2:]
start = time.time()
out2 =mfn_softmax_CUDA.mfn_forward(
    x,
    indices.to(torch.float32),
    model.linear_0.weight,
    model.linear_0.bias.squeeze(),
    model.non_linear_in_0.weight.squeeze(),
    model.non_linear_in_0.bias.squeeze(),
    model.out_linear.weight.squeeze(),
    model.out_linear.bias.squeeze())
torch.cuda.synchronize()
print('Custom Forward pass {:.3f}'.format(time.time()-start))
print(phi)
#print(out2/out2.sum(dim=1)[:,None])
print(torch.softmax(out2-out2.max(dim=1)[0].unsqueeze(-1), dim=1))
#print((model.linear_0.bias.squeeze()@ model.out_linear.weight.squeeze()) + model.out_linear.bias.squeeze())
#--------------------BATCH_SIZE 1--------------------#
#print("-------------model_out_grad ---------------")
#model_out_grad = grad_out @ torch.cat([w1,w2,w3,w4]).T
#print(grad_out @ torch.cat([w1,w2,w3,w4]).T)
#print(torch.autograd.grad(loss, phi, retain_graph=True)[0])
#--------------------BATCH_SIZE 1--------------------#
#out2  = out2*(w1+w2+w3+w4)
start = time.time()
grad_W1, grad_b1, grad_W2, grad_b2, grad_Wout, grad_bout =mfn_softmax_CUDA.mfn_backward(
    torch.autograd.grad(loss, phi, retain_graph=True)[0],
    out2-out2.max(dim=1)[0].unsqueeze(-1),
    x,
    indices.to(torch.float32),
    model.linear_0.weight,
    model.linear_0.bias.squeeze(),
    model.non_linear_in_0.weight.squeeze(),
    model.non_linear_in_0.bias.squeeze(),
    model.out_linear.weight.squeeze(),
    model.out_linear.bias.squeeze())
torch.cuda.synchronize()
print('Custom Backward pass {:.3f}'.format(time.time()-start))
#print(model.out_linear.bias.grad)
#print(grad_bout)

#Jacobian = mfn_unnorm_CUDA.mfn_jacobian(
#    out2,
#    x,
#    indices.to(torch.float32),
#    model.linear_0.weight,
#    model.linear_0.bias.squeeze(),
#    model.non_linear_in_0.weight.squeeze(),
#    model.non_linear_in_0.bias.squeeze(),
#    model.out_linear.weight.squeeze(),
#    model.out_linear.bias.squeeze()
#)
positional_weights = (torch.tensor([2**(i/2)*torch.pi for i in range(1,model.pos+1)])[:,None,None] * torch.eye(2).unsqueeze(0)).reshape(20,2).to('cuda')
x = data['coords'].squeeze().to('cuda')[:100,:]
start = time.time()
Jacobian = mfn_softmax_CUDA.mfn_positional_jacobian(
    out2-out2.max(dim=1)[0].unsqueeze(-1),
    x,
    indices.to(torch.float32),
    model.linear_0.weight,
    model.linear_0.bias.squeeze(),
    model.non_linear_in_0.weight.squeeze(),
    model.non_linear_in_0.bias.squeeze(),
    model.out_linear.weight.squeeze(),
    model.out_linear.bias.squeeze(),
    positional_weights
)
torch.cuda.synchronize()
print('Custom Jacobian {:.3f}'.format(time.time()-start))

start = time.time()
J = jacobian(model.forward,(x, indices.to(torch.float32)) )
torch.cuda.synchronize()
print('Jacobian {:.3f}'.format(time.time()-start))
