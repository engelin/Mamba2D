from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='wavefront_cuda',
    ext_modules=[
        CUDAExtension(
            name='wavefront_cuda',
            sources=[
                'wf_cuda_bind.cpp',
                'wf_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-lineinfo'
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)