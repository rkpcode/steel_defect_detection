from setuptools import setup, find_packages

setup(
    name="steel_defect_detection_system",
    version="0.0.1",
    description="Steel Defect Detection System using Computer Vision",
    author="rkpcode",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "Pillow>=8.0.0",
        "tensorflow>=2.10.0",
        "albumentations>=1.0.0",
        "tqdm",
        "kaggle",
    ],
    python_requires=">=3.8",
)
