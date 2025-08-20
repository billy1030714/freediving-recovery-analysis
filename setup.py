from setuptools import setup, find_packages

setup(
    name='hrr_project',
    version='0.1.0',
    description='A project for analyzing freediving heart rate recovery.',
    author='Su, Yi-Chuan',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.2.0',
        'shap>=0.40.0',
        'lime>=0.2.0',
        'joblib>=1.1.0',
        'tqdm>=4.62.0',
        'jinja2>=3.0.0',
    ],
)