from pathlib import Path

from setuptools import find_packages, setup

readme = Path(__file__).parent / 'asf_tools' / 'README.md'

setup(
    name='asf_tools',
    use_scm_version=True,
    description='Tools developed by ASF for working with SAR data',
    long_description=readme.read_text(),
    long_description_content_type='text/markdown',

    url='https://github.com/ASFHyP3/asf-tools',

    author='ASF APD/Tools Team',
    author_email='uaf-asf-apd@alaska.edu',

    license='BSD',
    include_package_data=True,

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    python_requires='~=3.8',

    install_requires=[
        'numpy',
        'gdal>=3.2.0',
    ],

    extras_require={
        'develop': [
            'flake8',
            'flake8-import-order',
            'flake8-blind-except',
            'flake8-builtins',
            'pytest',
            'pytest-cov',
            'pytest-console-scripts',
        ]
    },

    packages=find_packages(),

    entry_points={
        'console_scripts': [
            'make_composite = asf_tools.composite:main',
        ]
    },

    zip_safe=False,
)
