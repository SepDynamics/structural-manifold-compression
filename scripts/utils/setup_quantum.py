from setuptools import setup, Extension
import pybind11
import glob
import os

cpp_src_dir = "scripts/utils/sep_quantum_cpp"
sources = [
    os.path.join(cpp_src_dir, "sep_quantum.cpp"),
    os.path.join(cpp_src_dir, "qfh_impl.cpp"),
    os.path.join(cpp_src_dir, "sep_core/core/io_utils.cpp"),
    os.path.join(cpp_src_dir, "sep_core/core/manifold_builder.cpp"),
    os.path.join(cpp_src_dir, "sep_core/core/oanda_client.cpp"),
    os.path.join(cpp_src_dir, "sep_core/core/trading_signals.cpp"),
]
sources = list(set(sources))

ext_modules = [
    Extension(
        "sep_quantum",
        sources,
        include_dirs=[
            pybind11.get_include(),
            cpp_src_dir,
            os.path.join(cpp_src_dir, "sep_core"),
            os.path.join(cpp_src_dir, "sep_core/core"),
            os.path.join(cpp_src_dir, "sep_core/include"),
        ],
        extra_link_args=["-lhiredis", "-lcurl", "-lssl", "-lcrypto", "-lcpr", "-ltbb"],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-shared", "-std=c++17", "-fPIC"],
    ),
]

setup(
    name="sep_quantum",
    ext_modules=ext_modules,
)
