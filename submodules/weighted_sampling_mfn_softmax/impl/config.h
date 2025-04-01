#ifndef WEIGHTED_MFN_H_INCLUDED
#define WEIGHTED_MFN_H_INCLUDED

#define threads_pb 256 // threads per block


template <typename scalar_t>
__forceinline__ __device__ scalar_t unnormalize(
    scalar_t coord, 
    int size) 
{
    return ((coord + 1) / 2) * (size - 1);
}

// Clips coordinates to between 0 and clip_limit - 1
template<typename scalar_t>
__forceinline__ __device__ scalar_t clip_coordinates(
    scalar_t in, 
    int clip_limit) 
{
  return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
__forceinline__ __device__ scalar_t compute_index(
    scalar_t coord,
    int size)
{
  coord = unnormalize(coord, size);
  coord = clip_coordinates(coord, size);
  return coord;
}

__forceinline__ __device__ bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}


#endif
