from setuptools import setup, find_packages

setup(
    name="solenoid_tracker",
    version="0.1",
    packages=find_packages(), # Автоматически найдет core и physics
    install_requires=[
        "numpy",
        "numba",
        "matplotlib",
        "scipy",
        "tqdm"
    ],
)