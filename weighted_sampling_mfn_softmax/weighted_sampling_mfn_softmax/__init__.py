import torch 
import torch.nn as nn
from . import _C
from grid_indexing import indexing
import mfn_softmax_CUDA
import pdb

class _Kernel_based_sampler(torch.autograd.Function):
   
    
    @staticmethod
    def forward(
            ctx,
            input_, 
            grid,
            enc_grid,
            indices,
            positional_weights,
            W1,
            b1,
            W2,
            b2,
            Wout,
            bout
        ):
        mfn_args = (
            enc_grid, 
            indices,
            W1,
            b1,
            W2,
            b2,
            Wout,
            bout
        )

        weights = mfn_softmax_CUDA.mfn_forward(*mfn_args)
        weights = weights - weights.max(dim=1)[0].unsqueeze(-1)
        norm_weights = torch.softmax(weights, dim=1)
        sampler_args = (
            input_,
            grid, 
            norm_weights
        )
        output =  _C.mfn_sampler_forward(*sampler_args)
        ctx.save_for_backward(
            input_, 
            grid,
            enc_grid, 
            indices,
            weights,
            output,
            W1,
            b1,
            W2,
            b2,
            Wout,
            bout,
            positional_weights
        )
        return output

    
    @staticmethod
    def backward(
            ctx,
            grad_output,
        ):
        input_, grid, enc_grid, indices, weights, output, W1, b1, W2, b2, Wout, bout, positional_weights = ctx.saved_tensors
        jacob_args = (
            weights,
            grid, 
            indices,
            W1,
            b1,
            W2,
            b2,
            Wout,
            bout,
            positional_weights
        )
        weights_jacobian = mfn_softmax_CUDA.mfn_positional_jacobian(*jacob_args)
        norm_weights = torch.softmax(weights, dim=1)
        sampler_args = (
            grad_output,
            input_,
            grid, 
            norm_weights,
            weights_jacobian
        )
        grad_input, grad_grid, grad_weights =  _C.mfn_sampler_backward(*sampler_args)
        mfn_args = (
            grad_weights,
            weights,
            enc_grid, 
            indices,
            W1,
            b1,
            W2,
            b2,
            Wout,
            bout
        )
        grad_W1, grad_b1, grad_W2, grad_b2, grad_Wout, grad_bout = \
            mfn_softmax_CUDA.mfn_backward(*mfn_args)
        grads = (
            grad_input, 
            grad_grid,
            None,
            None,
            None,
            grad_W1,
            grad_b1,
            grad_W2,
            grad_b2,
            grad_Wout,
            grad_bout
        )
        return grads


def kernel_based_sampler(
        input_, 
        grid,
        mfn,
        index
        ):
        # mfn is an nn.Module defined as nn.ParameterList 
        W1 = mfn[0] 
        b1 = mfn[1] 
        W2 = mfn[2] 
        b2 = mfn[3] 
        Wout = mfn[4] 
        bout = mfn[5] 
        H, W = input_.shape[2:]
        indices = indexing(grid.squeeze().to('cpu'), H, W, 4).to('cuda')
        enc_grid, positional_weights = pos_encoding(grid.squeeze(),10)
        return _Kernel_based_sampler.apply(
            input_, 
            grid,
            enc_grid,
            indices.to(torch.float32),
            positional_weights,
            W1,
            b1,
            W2,
            b2,
            Wout,
            bout
            )

def pos_encoding(x,d):
      out = []
      out.append(x)
      for i in range(1,d+1):
          if i%2:
              out.append(torch.sin(2**(i/2)*torch.pi*x))
          else:
              out.append(torch.cos(2**(i/2)*torch.pi*x))
      weights = torch.tensor(
                [
                    2**(i/2)*torch.pi for i in range(1,d+1)
                ]
            )[:,None,None] * torch.eye(2).unsqueeze(0)
      positional_weights = weights.reshape(d*x.shape[-1],x.shape[-1]).to(x.device)
      return torch.cat(out, dim =1)[:,2:], positional_weights
