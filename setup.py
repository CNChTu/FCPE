from setuptools import setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='torchfcpe',
    description='The official Pytorch implementation of Fast Context-based Pitch Estimation (FCPE)',
    version='0.0.1',
    author='CNChTu',
    author_email='2921046558@qq.com',
    url='https://github.com/CNChTu/FCPE',
    install_requires=['einops', 'local_attention', 'torch', 'torchaudio', 'numpy'],
    packages=['torchfcpe'],
    package_data={'torchfcpe': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['pitch', 'audio', 'speech', 'music', 'pytorch', 'fcpe'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
