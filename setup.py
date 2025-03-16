from setuptools import setup, find_packages

setup(
    name="ExoVol",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'keras',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'optuna',
        'pymysql',
        'sqlalchemy',
        'requests',
        'alpaca-trade-api',
        'statsmodels',
        'scipy',
    ],
    author="premdev1234",
    author_email="b22bb047@iitj.ac.in",
    description="A volatility surface modeling and forecasting tool",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ExoVol",
)
