from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os

### very hacky way to get rid of "cc1plus: fatal error: cuda_runtime.h: No such file or directory" but nothing else worked :(
cuda_home = os.getenv("CONDA_PREFIX", "/usr/local/cuda")
print("\nSET CUDA_HOME TO: ", cuda_home)
include_dirs = [os.path.join(cuda_home, "targets/x86_64-linux/include"),
                os.path.join(cuda_home, "lib/python3.11/site-packages/nvidia/cuda_runtime/include")]

setup(
    name='pixelspointspolygons',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pixelspointspolygons.models.pointpillars.voxel_op',
            sources=[
                'pixelspointspolygons/models/pointpillars/voxelization/voxelization.cpp',
                'pixelspointspolygons/models/pointpillars/voxelization/voxelization_cpu.cpp',
                'pixelspointspolygons/models/pointpillars/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '--expt-relaxed-constexpr',
                    '--allow-unsupported-compiler'
                ],
            },
            include_dirs=include_dirs
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch', 'numpy'],
    zip_safe=False,
)
