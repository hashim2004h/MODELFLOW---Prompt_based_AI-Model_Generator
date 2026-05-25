"""
Package Setup Configuration
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='modelflow',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Prompt-Based AI Model Generator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/modelflow',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
    install_requires=[
        'flask>=3.0.0',
        'tensorflow>=2.15.0',
        'torch>=2.1.0',
        'autokeras>=1.1.0',
        'onnx>=1.15.0',
        'opencv-python>=4.8.1',
        'pillow>=10.1.0',
        'numpy>=1.24.3',
        'pandas>=2.1.3',
        'scikit-learn>=1.3.2',
        'openai>=1.3.5',
        'python-dotenv>=1.0.0',
        'pyyaml>=6.0.1',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'modelflow=run:main',
        ],
    },
)


