#include "backward.cuh"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>

namespace cg = cooperative_groups;

__global__ void mfn_backward_CUDA(
        TensorInfo<const float, int>(grad_output),
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(grad_W1),
        TensorInfo<float, int>(grad_b1),
        TensorInfo<float, int>(grad_W2),
        TensorInfo<float, int>(grad_b2),
        TensorInfo<float, int>(grad_Wout),
        TensorInfo<float, int>(grad_bout)
    )
{
    int N = input.sizes[0];
    int L = input.sizes[1];
    int W1_C = W1.sizes[0];
    auto thread = cg::this_grid().thread_rank();
    auto block = cg::this_thread_block();
    if (thread >= N)
        return;
    int thread_pos = (int)thread;
    glm::vec4 grad_out(\
        grad_output.data[4*thread_pos],\
        grad_output.data[4*thread_pos+1],\ 
        grad_output.data[4*thread_pos+2],\ 
        grad_output.data[4*thread_pos+3]);
    glm::vec4 out(\
        exp(output.data[4*thread_pos]),\
        exp(output.data[4*thread_pos+1]),\
        exp(output.data[4*thread_pos+2]),
        exp(output.data[4*thread_pos+3]));
    auto sum = static_cast<float>(0);
    sum = out.x + out.y + out.z + out.w;
    glm::mat4 grad_norm = glm::mat4(
        out.x*(sum -out.x)/pow(sum,2), (-out.x*out.y)/pow(sum,2), (-out.x*out.z)/pow(sum,2), (-out.x*out.w)/pow(sum,2),
        (-out.y*out.x)/pow(sum,2), out.y*(sum -out.y)/pow(sum,2), (-out.y*out.z)/pow(sum,2), (-out.y*out.w)/pow(sum,2),
        (-out.z*out.x)/pow(sum,2), (-out.z*out.y)/pow(sum,2), out.z*(sum -out.z)/pow(sum,2), (-out.z*out.w)/pow(sum,2), 
        (-out.w*out.x)/pow(sum,2), (-out.w*out.y)/pow(sum,2), (-out.w*out.z)/pow(sum,2), out.w*(sum -out.w)/pow(sum,2));
    glm::vec4 G = grad_out*glm::transpose(grad_norm);
    atomicAdd(grad_bout.data, (G.x+G.y+G.z+G.w));   
    for (int i = 0; i<W1_C; ++i){
        auto res = static_cast<float>(0);
        for (int j=0; j<L ; ++j){
            res += W1.data[L*i + j] * input.data[L*thread_pos+j];
        }
        res += b1.data[i];
        glm::vec4 inter(\ 
            sinf( W2.data[i]*indices.data[4*thread_pos] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+1] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+2] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+3] + b2.data[i]));
        glm::vec4 inter2(\ 
            Wout.data[i] * res * cosf( W2.data[i]*indices.data[4*thread_pos] + b2.data[i]),\
            Wout.data[i] * res * cosf( W2.data[i]*indices.data[4*thread_pos+1] + b2.data[i]),\
            Wout.data[i] * res * cosf( W2.data[i]*indices.data[4*thread_pos+2] + b2.data[i]),\
            Wout.data[i] * res * cosf( W2.data[i]*indices.data[4*thread_pos+3] + b2.data[i]));
        glm::vec4 inter3(\
            inter2.x*indices.data[4*thread_pos],\
            inter2.y*indices.data[4*thread_pos+1],\
            inter2.z*indices.data[4*thread_pos+2],\
            inter2.w*indices.data[4*thread_pos+3]);
        for (int k=0; k<L ; ++k){
            atomicAdd(&grad_W1.data[L*i+k], glm::dot(inter, G)*Wout.data[i]*input.data[L*thread_pos+k]);
        }
        atomicAdd(&grad_b1.data[i], glm::dot(inter, G)*Wout.data[i]);
        atomicAdd(&grad_Wout.data[i], glm::dot(inter, G)*res);
        atomicAdd(&grad_b2.data[i], glm::dot(inter2, G));
        atomicAdd(&grad_W2.data[i],glm::dot(inter3,G));
    }
}

void BACKWARD::mfn_backward(
        TensorInfo<const float, int>(grad_output),
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(grad_W1),
        TensorInfo<float, int>(grad_b1),
        TensorInfo<float, int>(grad_W2),
        TensorInfo<float, int>(grad_b2),
        TensorInfo<float, int>(grad_Wout),
        TensorInfo<float, int>(grad_bout)
    )
{
    int N = input.sizes[0];
    mfn_backward_CUDA<<<(N+threads_pb-1)/threads_pb, threads_pb >>>(
        grad_output,
        output,
        input,
        indices,
        W1,
        b1,
        W2,
        b2,
        Wout,
        bout,
        grad_W1,
        grad_b1,
        grad_W2,
        grad_b2,
        grad_Wout,
        grad_bout
    );
}
