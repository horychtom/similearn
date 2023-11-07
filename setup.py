from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="similearn",
    version="2.2.2",
    author="Tomas Horych",
    author_email="t.horych@media-bias-research.org",
    description="Embedding trianing framework",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://www.SBERT.net",
    download_url="https://github.com/horychtom/similearn",
    packages=find_packages(),
    python_requires=">=3.6.0",
    install_requires=[
        "transformers>=4.6.0,<5.0.0",
        "tqdm",
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "scikit-learn",
        "scipy",
        "nltk",
        "sentencepiece",
        "huggingface-hub>=0.4.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Transformer Networks BERT PEFT LoRA sentence embedding PyTorch NLP deep learning",
)
