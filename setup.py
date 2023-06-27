
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
]

setup_kwargs = {
    'name': 'alignments',
    'version': '0.0.18',
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

