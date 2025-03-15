from setuptools import setup, find_packages

setup(
    name="data-science-hack-functions",
    version="0.1.0",  # ✅ Will be updated automatically via GitHub Actions
    description="A collection of data science utility functions",
    author="Harshithan Kavitha Sukumar",
    author_email="harshithan.ks2002@gmail.com",
    url="https://github.com/Harshithan07/data_science_hack_functions",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "scipy", "tabulate"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
