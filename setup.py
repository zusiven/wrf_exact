from setuptools import setup, find_packages

setup(
    name='wrf_exact',          # pip 安装时的名字
    version='0.1.0',
    description='wrf自用提取工具',
    packages=find_packages(),
    python_requires='>=3.8',
)