try:
    # Use setuptools (i.e., an extension of distutils)  when available, e.g., because it provides support for
    # "develop" installs.
    from setuptools import setup
except ImportError:
    # Fallback to distutils in case setuptools is not available.
    from distutils.core import setup

__author__ = 'Yves callaert'

setup(
    name="dsmodels",
    description="dsmodels",
    version="0.0.0.1",
    packages=["dsmodels", "dsmodels.arima"],
    scripts=[],
    setup_requires=['numpy==1.14.0'],
    install_requires=[
        'six==1.11.0', 'numpy==1.14.0','pandas==0.21.0', 'scikit-learn==0.19.1','matplotlib', 'statsmodels==0.9.0','scipy==1.0.1'
    ],
    dependency_links=[
    ]
)
