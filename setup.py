from setuptools import setup

setup(name='pybullet_swarming',
    version='1.0.0',
    install_requires=[
        'wandb',
        'numpy',
        'matplotlib',
        'pybullet',
        'cflib',
        'cfclient',
        'scipy',
        'pthflops'
        ]
)
