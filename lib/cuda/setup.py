from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='render_utils_cuda',
      ext_modules=[CUDAExtension('render_utils', ['render_utils.cpp', 'render_utils_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension},
      verbose=True)

setup(name='total_variation_cuda',
      ext_modules=[CUDAExtension('total_variation', ['total_variation.cpp', 'total_variation_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension},
      verbose=True)

setup(name='adam_upd_cuda',
      ext_modules=[CUDAExtension('adam_upd', ['adam_upd.cpp', 'adam_upd_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension},
      verbose=True)
