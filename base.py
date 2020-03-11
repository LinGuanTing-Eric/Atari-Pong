import numpy as np
import keras
from keras.layers import Dense
from keras.layers.core import Flatten


def policy_network_model(number_of_actions):
    """
    bulid policy network.
    """
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=((80, 80, 1))))
    model.add(Dense(512, activation="relu", kernel_initializer="glorot_normal"))
    model.add(Dense(number_of_actions, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()

    return model


def rgb2gray(rgb):
    """
    convert rgb image into grayscale.
    """
    return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])


def preprocess_frame(next_frame, curr_frame):
    """
    convert image into grayscale.
    """
    # translate type into int32.
    n_frame = next_frame.astype(np.int32)
    c_frame = curr_frame.astype(np.int32)
    # remove backgound colors
    n_frame[np.logical_or(n_frame == 144, n_frame == 109)] = 0
    c_frame[np.logical_or(c_frame == 144, c_frame == 109)] = 0
    # Image difference.
    diff = n_frame - c_frame
    # Convert to grayscale.
    diff = rgb2gray(diff)
    # croping top and bottom of game screen and subsample by 2.
    diff = diff[35:195:2, ::2]
    # rescale numbers between 0 and 1.
    max_val = diff.max() if diff.max() > abs(diff.min()) else abs(diff.min())
    if max_val != 0:
        diff = diff / max_val
    return diff
