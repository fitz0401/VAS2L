from setuptools import setup, find_packages

setup(
    name='vas2l',
    version='0.1.0',
    packages=find_packages(),
    description='Official implementation of VAS2L: Vision-Action-Sound2Language modules',
    url='git@github.com:fitz0401/vas2l.git',
    author='ze fu',
    author_email='ze.fu@kuleuven.be',
    license='MIT',
    install_requires=[
        'typing',
        'typing_extensions',
    ],
    zip_safe=False
)