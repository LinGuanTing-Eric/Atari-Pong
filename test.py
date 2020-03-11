import sys, os
import gym
import numpy as np

import keras
from keras.layers import Dense
from keras.layers.core import Flatten

import base

# Setup model path
model_path = "models/" + sys.argv[1]

# Script Parameters
number_of_actions = 6

# Initialize OpenAI Gym environment
env = gym.make("Pong-v0")
next_obse = env.reset()
curr_obse = next_obse

# Bulid the policy neural network.
policy_network_model = base.policy_network_model(number_of_actions)

# Load weights
policy_network_model.load_weights(model_path)

while True:

    env.render()

    input_net = base.preprocess_frame(next_obse, curr_obse)
    curr_obse = next_obse

    # Predict probabilities from the Keras model.
    action_prob = policy_network_model.predict_on_batch(
        input_net.reshape(1, 80, 80, 1)
    )[0, :]
    action = np.random.choice(number_of_actions, p=action_prob)

    # Execute one action in the environment
    next_obse, reward, done, info = env.step(action)

    # One game is finished, so reset environment
    if done:
        obs = env.reset()
