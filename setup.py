import os
from pathlib import Path
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gpt',
    version='1.1.4',
    packages=['gpt'],
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'gpt = gpt.gpt:main',
        ],
    },
)

base_path = Path(os.path.expanduser("~/.gpt"))
base_path.mkdir(exist_ok=True)
print("Please put your .env file in the ~/.gpt/ folder")
