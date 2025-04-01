#include "jacobian.cuh"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>

namespace cg = cooperative_groups;

__global__ void mfn_jacobian_CUDA(
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(jacobian)
    )
{
    int N = input.sizes[0];
    int W1_H = W1.sizes[0];
    int W1_W = W1.sizes[1];
    int J_N = jacobian.strides[0];
    int J_H = jacobian.strides[1];
    auto thread = cg::this_grid().thread_rank();
    auto block = cg::this_thread_block();
    if (thread >= N)
        return;
    int thread_pos = (int)thread;
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
    for (int i = 0; i<W1_H; ++i){
        glm::vec4 inter(\ 
            sinf( W2.data[i]*indices.data[4*thread_pos] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+1] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+2] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+3] + b2.data[i]));
        auto res = static_cast<float>(0);
        for (int j=0; j<W1_W; ++j){
            glm::vec4 inter2(\ 
                Wout.data[i] * W1.data[W1_W*i + j] * inter.x, 
                Wout.data[i] * W1.data[W1_W*i + j] * inter.y, 
                Wout.data[i] * W1.data[W1_W*i + j] * inter.z, 
                Wout.data[i] * W1.data[W1_W*i + j] * inter.w);
            glm::vec4 G = glm::transpose(grad_norm)*inter2;
            atomicAdd(&jacobian.data[J_N*thread_pos + j], G.x);
            atomicAdd(&jacobian.data[J_N*thread_pos + j + J_H], G.y);
            atomicAdd(&jacobian.data[J_N*thread_pos + j + 2*J_H], G.z);
            atomicAdd(&jacobian.data[J_N*thread_pos + j + 3*J_H], G.w);
        }
    }
}

__global__ void mfn_positional_jacobian_CUDA(
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<const float, int>(weights),
        TensorInfo<float, int>(jacobian)
    )
{
    int N = input.sizes[0];
    int L = input.sizes[1];
    int W1_C = W1.sizes[0];
    int W1_W = W1.sizes[1];
    int J_N = jacobian.strides[0];
    int J_H = jacobian.strides[1];
    int w_H = weights.strides[0];
    auto thread = cg::this_grid().thread_rank();
    auto block = cg::this_thread_block();
    if (thread >= N)
        return;
    int thread_pos = (int)thread;
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
    for (int i = 0; i<W1_C; ++i){
        glm::vec4 inter(\ 
            sinf( W2.data[i]*indices.data[4*thread_pos] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+1] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+2] + b2.data[i]),\
            sinf( W2.data[i]*indices.data[4*thread_pos+3] + b2.data[i]));
        for (int k=0; k<L; ++k){
            auto res = static_cast<float>(0);
            for (int j=0; j<W1_W; ++j){
                float t0 = weights.data[j*w_H]*input.data[thread_pos*L] +\
                    weights.data[j*w_H+1]*input.data[thread_pos*L+1];
                float t = int(floorf(j/2))%2==0 ? cosf(t0) : -sinf(t0);
                res += W1.data[W1_W*i + j]*t*weights.data[j*w_H+k];
            }
            glm::vec4 inter2(\ 
                Wout.data[i] * res * inter.x, 
                Wout.data[i] * res * inter.y, 
                Wout.data[i] * res * inter.z, 
                Wout.data[i] * res * inter.w);
            glm::vec4 G = glm::transpose(grad_norm)*inter2;
            atomicAdd(&jacobian.data[J_N*thread_pos + k], G.x);
            atomicAdd(&jacobian.data[J_N*thread_pos + k + J_H], G.y);
            atomicAdd(&jacobian.data[J_N*thread_pos + k + 2*J_H], G.z);
            atomicAdd(&jacobian.data[J_N*thread_pos + k + 3*J_H], G.w);
        }
    }
}

void JACOBIAN::mfn_jacobian(
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(jacobian)
    )
{
    int N = input.sizes[0];
    mfn_jacobian_CUDA<<<(N+threads_pb-1)/threads_pb, threads_pb >>>(
        output,
        input,
        indices,
        W1,
        b1,
        W2,
        b2,
        Wout,
        bout,
        jacobian
    );
}


void JACOBIAN::mfn_positional_jacobian(
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<const float, int>(positional_weights),
        TensorInfo<float, int>(jacobian)
    )

{
    int N = input.sizes[0];
    mfn_positional_jacobian_CUDA<<<(N+threads_pb-1)/threads_pb, threads_pb >>>(
        output,
        input,
        indices,
        W1,
        b1,
        W2,
        b2,
        Wout,
        bout,
        positional_weights,
        jacobian
    );
}
