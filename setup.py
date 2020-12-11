from setuptools import setup

setup(name="GraVe",
      version="0.1.3",
      description="GraVe: Graph Vectors.",
      long_description="Graph vectors.",
      license="Apache License 2.0",
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
      ],
      url='http://github.com/lantunes/GraVe',
      author="Luis M. Antunes",
      author_email="lantunes@gmail.com",
      packages=["grave"],
      keywords=["graphs", "embeddings"],
      python_requires='>3.5.6',
      install_requires=["numpy >= 1.15.2", "autograd >= 1.3", "networkx >= 2.4", "scipy >= 1.1.0", "tqdm >= 4.54.1"])
