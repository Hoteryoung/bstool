import os
import glob
import platform
import subprocess
import time
import numpy as np
from os import path
from setuptools import find_packages, setup, Extension

from Cython.Build import cythonize
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 0
MINOR = 0
PATCH = 1
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = 'bstool/version.py'


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from bstool.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}

__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
# def get_cuda_extensions():
#     this_dir = path.dirname(path.abspath(__file__))
#     extensions_dir = path.join(this_dir, "bstool", "csrc")

#     main_source = path.join(extensions_dir, "vision.cpp")
#     sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))
#     source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
#         path.join(extensions_dir, "*.cu")
#     )

#     sources = [main_source] + sources
#     extension = CppExtension

#     extra_compile_args = {"cxx": []}
#     define_macros = []

#     if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
#         extension = CUDAExtension
#         sources += source_cuda
#         define_macros += [("WITH_CUDA", None)]
#         extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]

#         # It's better if pytorch can do this by default ..
#         CC = os.environ.get("CC", None)
#         if CC is not None:
#             extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

#     include_dirs = [extensions_dir]

#     ext_modules = [
#         extension(
#             "bstool._C",
#             sources,
#             include_dirs=include_dirs,
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#     ]

#     return ext_modules

def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }
    print(name, module, sources)
    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension

if __name__ == '__main__':
    write_version_py()
    setup(
        name='bstool',
        version=get_version(),
        description='Tools for jwwangchn Research',
        # long_description=readme(),
        keywords='computer vision, instance segmentation',
        url='https://github.com/jwwangchn/bstool',
        packages=find_packages(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=['cython', 'numpy'],
        # tests_require=['pytest'],
        install_requires=['numpy', 'matplotlib', 'six', 'terminaltables',
            'pycocotools', 'shapely', 'geojson', 'scikit-image', 'geopandas', 'rasterio', 'networkx'
        ],
        ext_modules=[
            make_cuda_ext(
                name='rnms_ext',
                module='bstool.csrc.nms',
                sources=['src/rnms_ext.cpp', 'src/rcpu/rnms_cpu.cpp']),
            make_cuda_ext(
                name='rbbox_geo_cuda',
                module='bstool.csrc.rbbox_geo',
                sources=[],
                sources_cuda=[
                    'src/rbbox_geo_cuda.cpp', 'src/rbbox_geo_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
