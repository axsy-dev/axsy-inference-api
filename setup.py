from setuptools import find_packages, setup

setup(
    name="axsy_detection",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "argparse",
        "numpy",
        "torch",
    ],
    extras_require={"gcs": ["google-cloud-storage"]},
    entry_points={
        "console_scripts": [
            "axsy_detection=axsy_detection:run",
        ],
    },
    author="Joseph Hills",
    author_email="joseph.hills@axsy.com",
    url="https://github.com/axsy-dev/axsy-notation",
    description="Run Axsy detection  on images in a single directory",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 3 - Alpha",
    ],
)
