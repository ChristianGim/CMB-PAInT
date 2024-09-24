from setuptools import setup, find_namespace_packages
from CMBPAInT.__version import __version__

setup(
    name='CMBPAInT',
    version=__version__,
    description='A Python package to inpaint Cosmic Microwave Background (CMB) intensity and polarization maps.',
    license='MIT License',
    license_files=['LICENSE'],
    author="Gimeno-Amo, Christian",
    author_email='gimenoc@ifca.unican.es',
    packages=['CMBPAInT','CMBPAInT.scripts','CMBPAInT.Tools'],
    package_dir={"CMBPAInT": "CMBPAInT"},
    url='https://https://github.com/ChristianGim/CMB-PAInT',
    keywords='inpainting,anisotropy,healpy,polarization,cmb,healpix,cosmology,maps,sampling',
    install_requires=[
          'numpy',
          'scipy',
          'healpy',
          'tqdm',
          'psutil',
          'dask'
      ],

)
