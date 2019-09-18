from setuptools import setup, find_packages

REQUIRED = ['tensorflow', 'gym', 'numpy', 'pandas', 'matplotlib', 'ray[rllib]']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sumo-rl',
    version='0.3',
    packages=['sumo_rl',],
    install_requires=REQUIRED,
    author='Roman',
    author_email='c.d.roman@warwick.ac.uk',
    long_description=long_description,
    url='https://github.com/maunhb/sumo-marl',
    license="MIT",
    description='Environments inheriting OpenAI Gym Env and RL algorithms to control Traffic Signal controllers on SUMO.'
)
