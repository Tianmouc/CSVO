import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='csvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['csvo/altcorr/correlation.cpp', 'csvo/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba',
            sources=['csvo/fastba/ba.cpp', 'csvo/fastba/ba_cuda.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'csvo/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')],
            sources=[
                'csvo/lietorch/src/lietorch.cpp', 
                'csvo/lietorch/src/lietorch_gpu.cu',
                'csvo/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

