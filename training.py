import gym
import numpy as np

import keras
from keras.layers.core import Flatten
from keras.layers import Dense

import base


def discount_reward(r, gamma=0.9):
    """
    let previous action which get 0 reward obtaion non-zero reward.
    """
    discounted_r = np.zeros_like(r)  # copy format of r-list which value all zeros.
    disc_factor = 0
    for t in range(r.size - 1, 0, -1):
        if r[t] != 0:
            disc_factor = 0
        disc_factor = disc_factor * gamma + r[t]
        discounted_r[t] = disc_factor
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


# Script Parameters
learning_rate = 0.1
render = False
number_of_actions = 6

states, action_prob_grads, rewards, action_probs = [], [], [], []
reward_sums = []  # Used to calculate mean reward sum.
reward_sum = 0  # Open logfile
episode_number = 0

# Initialize
env = gym.make("Pong-v0")
curr_obse = env.reset()
next_obse = curr_obse

policy_network_model = base.policy_network_model(number_of_actions)

while True:

    if render:
        env.render()

    net_input = base.preprocess_frame(next_obse, curr_obse)
    curr_obse = next_obse

    # Predict probabilies form the Keras model and sample action.
    action_prob = policy_network_model.predict_on_batch(
        net_input.reshape(1, 80, 80, 1)
    )[0, :]
    action = np.random.choice(number_of_actions, p=action_prob)

    # Execute one action in the environment.
    next_obse, reward, done, info = env.step(action)
    reward_sum += reward

    # Remerber what we need for training the model.
    states.append(net_input)
    action_probs.append(action_prob)
    rewards.append(reward)

    # Also remeber gradient of the action probabilities.
    y = np.zeros(number_of_actions)
    y[action] = 1
    action_prob_grads.append(y - action_prob)

    if done:
        # One game is complete (i.e. one of the players has gotten 21 points).
        # Time to train the policy network
        episode_number += 1

        # Remerber last 40 reward sums to calculate mean reward sum.
        reward_sums.append(reward_sum)
        if len(reward_sums) > 40:
            reward_sums.pop(0)

        # Print the current performance of the agent ...
        s = "Episode %d Total Episode Reward: %f , Mean %f" % (
            episode_number,
            reward_sum,
            np.mean(reward_sums),
        )
        print(s)

        # Save model weights.
        policy_network_model.save_weights("models/Reward:%d.h5" % (int(reward_sum)))

        # Propagate the rewards back to actions where no reward was given.
        # Rewards for earlier actions are attenuated
        rewards = np.vstack(rewards)
        actions_prob_grads = np.vstack(action_prob_grads)
        rewards = discount_reward(rewards)

        # Accumulate observed states, calculate updated action probabilities.
        # Let y-target data which close to we want.
        x = np.vstack(states).reshape(-1, 80, 80, 1)
        y = action_probs + learning_rate * rewards * action_prob_grads

        # Train policy network.
        # x-numpy array of training data.
        # y-numpy array of target data.
        policy_network_model.train_on_batch(x, y)

        # Initial everything for the next game
        states, action_prob_grads, rewards, action_probs = [], [], [], []
        reward_sum = 0
        obs = env.reset()
