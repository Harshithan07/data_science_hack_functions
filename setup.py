from setuptools import setup, find_packages

setup(
    name="data-science-hack-functions",
    version="0.2.0",  # 🔁 New version
    description="A collection of data science utility functions for EDA, preprocessing, evaluation, and tuning.",
    author="Harshithan Kavitha Sukumar",
    author_email="harshithan.ks2002@gmail.com",
    url="https://github.com/Harshithan07/data_science_hack_functions",
    packages=find_packages(include=["data_science_hack_functions", "data_science_hack_functions.*"]),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "tabulate",
        "scikit-learn>=1.0",
        "optuna>=3.0",
        "matplotlib",
        "seaborn",
        "rich"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
