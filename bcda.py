
import tensorflow as tf
from tensorflow import keras 
from keras import layers

import numpy as np
import random
from sklearn import preprocessing
from scipy.linalg import pinv
from scipy.special import expit as sigmoid
from collections import deque
import gym
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import time
from scipy import linalg as LA

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from noise import OUActionNoise
from buffer import Buffer as Buffer

from BLSCriticNetwork import BLSCriticNetwork as bls



# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))



def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


def update_actor(actor_network, critic_network, states, optimizer):

  """
  Update the actor network using gradients derived from the critic network.

  Parameters:
  - actor_network: PyTorch model for the actor.
  - critic_network: BLSCriticNetwork instance for the critic.
  - states: Current batch of states.
  - optimizer: Optimizer for the actor network.
  """
  # Ensure states are in tensor format
  states_tensor = torch.tensor(states, dtype=torch.float32)

  # Zero gradients for the optimizer
  optimizer.zero_grad()

  # Generate actions from the actor network
  actions = actor_network(states_tensor)

  # IMPORTANT: Convert actions to numpy for critic prediction, then back to tensor
  # Ensure actions are detached when used for critic prediction to avoid computing gradients for the critic
  Q_values = critic_network.predict(states, actions.cpu().detach().numpy())

  # Convert Q_values back to tensor and ensure they require gradients
  Q_values_tensor = torch.tensor(Q_values, dtype=torch.float32, requires_grad=True)

  # Calculate loss as negative Q-values (to maximize them)
  actor_loss = -torch.mean(Q_values_tensor)

  # Backpropagate gradients
  actor_loss.backward()

  # Update actor parameters
  optimizer.step()

if __name__ == '__main__':

    problem = "Pendulum-v1"
    env = gym.make(problem)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    total_episodes = 100
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005

    buffer = Buffer(num_states=num_states, num_actions=num_actions, actor=actor_model, critic=critic_model, target_actor=target_actor, target_critic=target_critic, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, gamma=gamma, buffer_capacity=100000, batch_size=64)



    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Takes about 4 min to train
    for ep in range(50):

        #prev_state = env.reset()
        prev_state, _ = env.reset()
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)
            # Recieve state and reward from environment.
            #state, reward, done, info = env.step(action)
            state, reward, terminated, truncated, info = env.step(action) # changed


            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # End this episode when `done` is True
            if terminated or truncated:
            #if done:1
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()