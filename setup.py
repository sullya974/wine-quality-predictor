import setuptools


setuptools.setup(
    name='wqp',
    version='1.0.0',
    author='webmestre.gerard@gmail.com',
    description='Wine quality predictor - a package machine learning algo to predict wine quality'
    packages=setuptools.find-packages(),
    install_requires=[
        "scikit-learn==0.22.1",
        "pandas==1.0.1"
    ]
)        
