from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os


setup(
    name='weighted_sampling_mfn_softmax',
    packages=['weighted_sampling_mfn_softmax'],
    ext_modules=[
        CUDAExtension(
            name='weighted_sampling_mfn_softmax._C',
            sources=[
            'mfn_weighted_sampler.cu',
            'impl/forward.cu',
            'impl/backward.cu',
            'ext.cpp']
        )    
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
