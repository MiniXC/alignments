
# -*- coding: utf-8 -*-
from setuptools import setup

long_description = None
INSTALL_REQUIRES = [
    'click>=8.0.4',
    'rich>=10.11.0',
]
ENTRY_POINTS = {
    'console_scripts': [
        'alignments = alignments:cli',
    ],
}

setup_kwargs = {
    'name': 'alignments',
    'version': '0.0.1',
    'description': '',
    'long_description': long_description,
    'license': 'MIT',
    'author': '',
    'author_email': 'Christoph Minixhofer <christoph.minixhofer@gmail.com>',
    'maintainer': None,
    'maintainer_email': None,
    'url': '',
    'packages': [
        'alignments',
    ],
    'package_dir': {'': 'src'},
    'package_data': {'': ['*']},
    'install_requires': INSTALL_REQUIRES,
    'python_requires': '>=3.6',
    'entry_points': ENTRY_POINTS,

}


setup(**setup_kwargs)
