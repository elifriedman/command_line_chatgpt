from pathlib import Path
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gpt',
    version='1.0',
    packages=['src'],
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'gpt = src.gpt:main',
        ],
    },
)

base_path = Path(os.path.expanduser("~/.gpt"))
base_path.mkdir(exist_ok=True)
print("Please put your .env file in the ~/.gpt/ folder")
