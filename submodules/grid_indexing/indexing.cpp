#include "indexing.h" //always in the beginning in order to use full namespace qualifier
#include <torch/extension.h>
#include <iostream>
//#include <ATen/Parallel.h>
#include <c10/util/irange.h>
#ifndef _OPENMP
#define _OPENMP
#else
#include <omp.h>
#endif
#include <ATen/ParallelOpenMP.h>


torch::Tensor indexing(
    const torch::Tensor& input,
    const int64_t H,
    const int64_t W, 
    const int grain_size_
    )
{
    int num_threads = omp_get_max_threads();
    //std::cout << "Number of threads: " << num_threads << std::endl;
    int N = input.size(0);
    auto indices = torch::empty({N,4}, input.options().dtype(torch::kInt64));
    auto inp_ptr = input.contiguous().data_ptr<float>();
    auto ind_ptr = indices.data_ptr<int64_t>();
    at::parallel_for(0, N, grain_size_, [&](int start, int end) {
    //printf("%d, %d\n" , start, end);
    //std::cout << "Processing chunk from " << start << " to " << end <<"thread"<<omp_get_thread_num()<< std::endl;
    for (const auto n : c10::irange(start, end)) 
    {
       //std::cout <<"N:"<<N<<','<<"start:"<<start<<','<<"end"<<end<<','<<"n"<<n<<std::endl;
       float x = *(inp_ptr+2*n);
       float y = *(inp_ptr+2*n+1);//inp_ptrCoor];
       float ix = grid_sampler_compute_source_index(x, W);
       float iy = grid_sampler_compute_source_index(y, H);

       // get corner pixel values from (x, y)
       // for 4d, we use north-east-south-west
       int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
       int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
       ind_ptr[4*n] = (ix_nw) + (iy_nw)*W;
       ind_ptr[4*n+1] = (ix_nw+1) + (iy_nw)*W;
       ind_ptr[4*n+2] = (ix_nw) + (iy_nw+1)*W;
       ind_ptr[4*n+3] = (ix_nw+1) + (iy_nw+1)*W;
      }
    });
    

    return indices;
}


