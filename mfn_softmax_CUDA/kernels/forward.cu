#include "forward.cuh"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void mfn_forward_CUDA(
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(output)
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
    for (int i = 0; i<W1_C; ++i){
        auto res = static_cast<float>(0);
        for (int j=0; j<L ; ++j){
            res += W1.data[L*i + j] * input.data[L*thread_pos+j];
        }
        res += b1.data[i];
        output.data[4*thread_pos] += Wout.data[i]*(    \
            res * sinf( W2.data[i]*indices.data[4*thread_pos] + b2.data[i]) \
            );
        output.data[4*thread_pos+1] += Wout.data[i]*(    \
            res * sinf( W2.data[i]*indices.data[4*thread_pos+1] + b2.data[i]) \
            );
        output.data[4*thread_pos+2] += Wout.data[i]*(    \
            res * sinf( W2.data[i]*indices.data[4*thread_pos+2] + b2.data[i]) \
            );
        output.data[4*thread_pos+3] += Wout.data[i]*(    \
            res * sinf( W2.data[i]*indices.data[4*thread_pos+3] + b2.data[i]) \
            );
    }
    output.data[4*thread_pos] += bout.data[0];
    output.data[4*thread_pos+1] += bout.data[0];
    output.data[4*thread_pos+2] += bout.data[0];
    output.data[4*thread_pos+3] += bout.data[0];
}

//__global__ void normalization(
//        TensorInfo<float, int>(output)
//    )
//{
//    int N = output.sizes[0];
//    auto thread = cg::this_grid().thread_rank();
//    auto block = cg::this_thread_block();
//    if (thread >= N)
//        return;
//    int thread_pos = (int)thread;
//    float sum = output.data[4*thread_pos] + \
//        output.data[4*thread_pos+1] + \ 
//        output.data[4*thread_pos+2] + \
//        output.data[4*thread_pos+3];
//    if (sum!=0){
//        output.data[4*thread_pos] /= sum;
//        output.data[4*thread_pos+1] /= sum;
//        output.data[4*thread_pos+2] /= sum;
//        output.data[4*thread_pos+3] /= sum;
//    }
//}

void FORWARD::mfn_forward(
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(output)
    )
{
    int N = input.sizes[0];
    mfn_forward_CUDA<<<(N+threads_pb-1)/threads_pb, threads_pb >>>(
        input,
        indices,
        W1,
        b1,
        W2,
        b2,
        Wout,
        bout,
        output
    );
    //normalization<<<(N+threads_pb-1)/threads_pb, threads_pb >>>(
    //    output
    //);
}
