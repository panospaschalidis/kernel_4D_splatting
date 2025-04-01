#include <torch/extension.h>
#include <iostream>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>
#include <omp.h> // Include the OpenMP header
std::mutex print_mutex;
torch::Tensor mfn_forward(torch::Tensor z, int64_t grain_size) {
    torch::Tensor z_out = at::empty({z.size(0), z.size(1)}, z.options());
    int64_t batch_size = z.size(0);

    // Print the number of threads
    int num_threads = omp_get_max_threads();
    std::cout << "Number of threads: " << num_threads << std::endl;

    at::parallel_for(0, batch_size, grain_size, [&](int64_t start, int64_t end) {
        std::lock_guard<std::mutex> guard(print_mutex);
        std::cout << "Processing chunk from " << start << " to " << end 
                  << " by thread " << omp_get_thread_num() << std::endl;
        for (int64_t b = start; b < end; b++) {
            z_out[b] = z[b] * z[b];
        }
    });

    return z_out;
}


