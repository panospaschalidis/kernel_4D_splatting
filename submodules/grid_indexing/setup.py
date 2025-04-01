from setuptools import setup, Extension
from torch.utils import cpp_extension
#from torch.utils.cpp_extension  import CUDAExtension, BuildExtension                 

setup(          
    name='grid_indexing',
    #packages=['linear_custom'],
    ext_modules=[
        cpp_extension.CppExtension(
            name = 'grid_indexing', 
            sources = [
                'indexing.cpp',
                'ext.cpp'
            ],
            extra_compile_args=['-fopenmp'],
            #extra_link_args=['-lgomp']
        )       
    ],          
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
#setup(
#    name='linear_custom',
#    packages=['linear_custom'],
#    ext_modules=[
#        CUDAExtension(
#            name = 'linear_custom._C', 
#            sources = [
#                'linear.cu', 
#                'ext.cpp'
#            ]
#        )
#    ],
#    cmdclass={
#        'build_ext': BuildExtension
#    }
#)
