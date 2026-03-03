from setuptools import setup, find_packages

setup(
    name="human-aligned-ai",
    version="0.1.0",
    description="End-to-End RLHF Pipeline for Human-Aligned AI",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
)
