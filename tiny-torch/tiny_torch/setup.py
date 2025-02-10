from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tiny-torch-nevmenko",
    version="0.1.0",
    author="-",
    author_email="-",
    description="Tiny Torch - Custom CUDA Extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["tiny_torch"],
    ext_modules=[
        CUDAExtension(
            "tiny_torch._C",
            sources=[
                "tiny_torch/cpp/relu.cpp",
                "tiny_torch/cpp/relu_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
    python_requires=">=3.7",
)
