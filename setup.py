
# -*- coding: utf-8 -*-
from setuptools import setup

long_description = None
INSTALL_REQUIRES = [
    'click>=8.0.4',
    'rich>=10.11.0',
    'torch>=1.4.0',
    'numpy>=1.19.5',
    'tqdm>=4.64.0',
    'textgrid>=1.5',
    'tgt>=1.5',
    'transformers>=4.5.1',
    "librosa>=0.9.2",
    "soundfile>=0.12.1",
    "torchaudio>=0.9.0",
]

setup_kwargs = {
    'name': 'alignments',
    'version': '0.1.8',
    'description': '',
    'long_description': long_description,
    'license': 'MIT',
    'author': '',
    'author_email': 'Christoph Minixhofer <christoph.minixhofer@gmail.com>',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/MiniXC/alignments',
    'packages': [
        'alignments',
    ],
    'package_data': {'': ['*']},
    'install_requires': INSTALL_REQUIRES,
    'python_requires': '>=3.6',

}


setup(**setup_kwargs)

