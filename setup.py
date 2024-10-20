from setuptools import setup

package_name = 'emptyseats'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/navigation_launch.py']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='redha',
    maintainer_email='m.reza.ramezani8@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emptyseats = emptyseats.main:main',
        ],
    },
)
