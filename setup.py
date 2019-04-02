from setuptools import setup


setup(
    name='sn_metrics',
    version='0.1',
    description='Metrics for supernovae',
    url='http://github.com/lsstdesc/sn_metrics',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    packages=['metrics', 'plotters'],
    python_requires='>=3.5',
    zip_safe=False,
    install_requires=[
        'sn_tools>=0.1',
        'sn_stackers>=0.1'
    ],
    dependency_links=[
        'git+https://github.com/lsstdesc/sn_tools.git@dev#egg=sn_tools-0.1',
        'git+https://github.com/lsstdesc/sn_stackers.git@dev#egg=sn_stackers-0.1',
    ],

)
