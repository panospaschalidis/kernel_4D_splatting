from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name='mfn_softmax_CUDA',
    ext_modules=[
        CUDAExtension(
            name='mfn_softmax_CUDA',
            sources=[
            'mfn.cu',
            'kernels/forward.cu',
            'kernels/backward.cu',
            'kernels/jacobian.cu',
            'ext.cpp'],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
        )    
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
