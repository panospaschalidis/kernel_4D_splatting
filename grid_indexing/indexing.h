#include <torch/extension.h>
#include <iostream>

template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize(scalar_t coord, int64_t size)
{
    return ((coord + 1) / 2) * (size - 1);
};

template<typename scalar_t>
static inline scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
  return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
};

template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int64_t size
    ) {
  coord = grid_sampler_unnormalize(coord, size);
  coord = clip_coordinates(coord, size);
  return coord;
};


torch::Tensor indexing(
    const torch::Tensor& input,
    const int64_t H,
    const int64_t W,
    const int grain_size_
);


