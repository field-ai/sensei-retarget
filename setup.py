from setuptools import setup, find_packages

setup(
    name="sensei-humanoid-retarget",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require={
        # Phase 1: GMR + mink IK
        "gmr": [
            "mink",
            "mujoco",
            "qpsolvers[daqp,proxqp]",
            "smplx @ git+https://github.com/vchoutas/smplx",
            "torch",
        ],
        # Phase 3: alpaqa PANOC solver
        # Install via: pip install -e third_party/alpaqa
        "alpaqa": [],
        # Dev
        "dev": [
            "pytest",
            "ruff",
        ],
    },
    author="",
    description="Modular humanoid motion retargeting pipeline platform",
    url="https://github.com/jay-fai/sensei-humanoid-retarget",
    license="MIT",
)
