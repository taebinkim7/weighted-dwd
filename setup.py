from setuptools import setup, find_packages

install_requires = ['numpy',
                    'cvxpy',
                    'scikit-learn',
                    'matplotlib'
                    ]

setup(name='wdwd',
      version='0.0.1',
      description='Code to reproduce Weighted Distance Weighted Discrimination',
      author='Taebin Kim',
      author_email='taebinkim@unc.edu',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=install_requires,
      zip_safe=False)
