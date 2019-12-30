import numpy as np
import matplotlib.pyplot as plt


def estimate_position(towers, z):
    pass


def estimate_motion(towers, z):
    pass

if __name__ == '__main__':
    # load the data
    data = np.load('./data_position.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z'] * np.pi/180

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    estimate_position(towers, z)

    # load the data
    data = np.load('./data_motion.npz')
    # towers
    towers = data['towers']
    # measurements
    z = data['z']

    print('Towers:', towers.shape)
    print('Measurements:', z.shape)

    estimate_motion(towers, z)
