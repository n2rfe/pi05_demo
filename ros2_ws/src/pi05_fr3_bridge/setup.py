from setuptools import setup
setup(
    name='pi05_fr3_bridge',
    version='0.0.1',
    packages=['pi05_fr3_bridge'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'bridge_node = pi05_fr3_bridge.bridge_node:main',
        ],
    },
)
