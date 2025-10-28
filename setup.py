from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


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
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch', 'numpy'],
    zip_safe=False,
)
