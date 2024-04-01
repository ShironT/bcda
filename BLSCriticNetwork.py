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


import tensorflow as tf
from tensorflow import keras
from keras import layers


class BLSCriticNetwork:
  
  def __init__(self, state_dim, action_dim, batch_size, n_fm=5, n_en=20, s=1, reg=1, tau=0.005):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.Q_target_dim = action_dim
    self.batch_size = batch_size
    self.n_fm = n_fm
    self.n_en = n_en
    self.s = s
    self.reg = reg
    self.tau = 0.005

    # Weights of the online network
    self.betaOfEachWindow = [] # a
    self.distOfMaxAndMin = [] # b
    self.minOfEachWindow = [] # c
    self.weightOfEnhanceLayer = [] # d
    self.parameterOfShrink = [] # e
    self.OutputWeight = [] # f

    # Weights of the target network
    self.a_t = []
    self.b_t = []
    self.c_t = []
    self.d_t = []
    self.e_t = []
    self.f_t = []

  def tansig(self, x):
    return (2/(1+np.exp(-2*x)))-1

  def shrinkage(self, a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z

  def sparse_bls(self, A, b):

    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = self.shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


  def calculateLayerWeights(self, x, L, N1, N2, N3):

    FeatureOfInputDataWithBias = np.hstack([x, 0.1 * np.ones((x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([x.shape[0], N2 * N1])

    random.seed(1)
    weightOfEachWindow = 2 * np.random.randn(x.shape[1] + 1, N1) - 1
    FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
    scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
    FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
    self.betaOfEachWindow = self.sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
    outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, self.betaOfEachWindow)
    self.distOfMaxAndMin = np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0)
    self.minOfEachWindow = np.min(outputOfEachWindow, axis=0)
    outputOfEachWindow = (outputOfEachWindow - self.minOfEachWindow) / self.distOfMaxAndMin
    OutputOfFeatureMappingLayer[:, N1 * 0:N1 * (0 + 1)] = outputOfEachWindow

    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    if N1 * N2 >= N3:
        random.seed(67797325)
        self.weightOfEnhanceLayer = LA.orth(2 * np.random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        self.weightOfEnhanceLayer = LA.orth(2 * np.random.randn(N2 * N1 + 1, N3).T - 1).T
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, self.weightOfEnhanceLayer)
    self.parameterOfShrink = self.s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = self.tansig(tempOfOutputOfEnhanceLayer * self.parameterOfShrink)
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])

    return InputOfOutputLayer


  def train(self, states, actions, Q_targets, init=False):
    L = 5
    N1 = self.n_fm
    N2 = 1 # No. of windows = 1
    N3 = self.n_en

    if np.ndim(states) == 1:
      state = state.reshape(1, -1)

    if np.ndim(actions) == 1:
      action = action.reshape(1, -1)


    #x = preprocessing.scale(np.hstack([states, actions]), axis=1)
    x = np.hstack([states, actions])
    y = Q_targets

    # Variable InputOfOutputLayer is taken from thecalculateLayerWeights function
    InputOfOutputLayer = self.calculateLayerWeights(x, L, N1, N2, N3)

    pinvOfInput = pinv(InputOfOutputLayer, self.reg)
    self.OutputWeight = np.dot(pinvOfInput, y)



    # For testing purpose
    return self.betaOfEachWindow, self.distOfMaxAndMin, self.minOfEachWindow, self.weightOfEnhanceLayer, self.parameterOfShrink, self.OutputWeight


  def predict(self, state, action):

    if np.ndim(state) == 1:
      state = state.reshape(1, -1)

    if np.ndim(action) == 1:
      action = action.reshape(1, -1)

    #test_x = preprocessing.scale(np.hstack([state, action]), axis=1)
    test_x = np.hstack([state, action])

    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], self.n_fm])
    outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, self.betaOfEachWindow)
    OutputOfFeatureMappingLayerTest[:, self.n_fm * 0:self.n_fm * (0 + 1)] = (outputOfEachWindowTest - self.minOfEachWindow) / self.distOfMaxAndMin
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, self.weightOfEnhanceLayer)
    OutputOfEnhanceLayerTest = self.tansig(tempOfOutputOfEnhanceLayerTest * self.parameterOfShrink)
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    OutputOfTest = np.dot(InputOfOutputLayerTest, self.OutputWeight)

    return OutputOfTest

  def updateTargetParam(self, tau):

    ### Cann this function as soon as training is finished
    # Update target parameters
    self.a_t = (1 - tau) * np.array(self.a_t) + tau * np.array(self.betaOfEachWindow)
    self.b_t = (1 - tau) * np.array(self.b_t) + tau * np.array(self.distOfMaxAndMin)
    self.c_t = (1 - tau) * np.array(self.c_t) + tau * np.array(self.minOfEachWindow)
    self.d_t = (1 - tau) * np.array(self.d_t) + tau * np.array(self.weightOfEnhanceLayer)
    self.e_t = (1 - tau) * self.e_t + tau * self.parameterOfShrink
    self.f_t = (1 - tau) * np.array(self.f_t + tau) * np.array(self.OutputWeight)

    return self.a_t, self.b_t, self.c_t, self.d_t, self.e_t, self.f_t


  def predictTarget(self, new_states, new_actions, init=False):

    # init=True if critic network is not yet trained. (Only in the first iteration)

    L = 5
    N1 = self.n_fm
    N2 = 1 # No. of windows = 1
    N3 = self.n_en

    # Check if 2D, otherwise convert into 2D
    if np.ndim(new_states) == 1:
      state = state.reshape(1, -1)

    if np.ndim(new_actions) == 1:
      action = action.reshape(1, -1)

    #x = preprocessing.scale(np.hstack([new_states, new_actions]), axis=1)
    x = np.hstack([new_states, new_actions])


    if init == True: # Initialization at the start of the loop

      InputOfOutputLayerTarget = self.calculateLayerWeights(x, L, N1, N2, N3)
      RandomOutputWeight = np.random.rand(N1+N3, 1) # Randomly generate this
      OutputOfTarget = np.dot(InputOfOutputLayerTarget, RandomOutputWeight)

      # assign initial target parameters
      self.a_t = self.betaOfEachWindow
      self.b_t = self.distOfMaxAndMin
      self.c_t = self.minOfEachWindow
      self.d_t = self.weightOfEnhanceLayer
      self.e_t = self.parameterOfShrink
      self.f_t = RandomOutputWeight


    else:
      FeatureOfInputDataWithBiasTarget = np.hstack([x, 0.1 * np.ones((x.shape[0], 1))])
      OutputOfFeatureMappingLayerTarget = np.zeros([x.shape[0], self.n_fm])
      outputOfEachWindowTarget = np.dot(FeatureOfInputDataWithBiasTarget, self.a_t)
      OutputOfFeatureMappingLayerTarget[:, self.n_fm * 0:self.n_fm * (0 + 1)] = (outputOfEachWindowTarget - self.c_t) / self.b_t
      InputOfEnhanceLayerWithBiasTarget = np.hstack([OutputOfFeatureMappingLayerTarget, 0.1 * np.ones((OutputOfFeatureMappingLayerTarget.shape[0], 1))])
      tempOfOutputOfEnhanceLayerTarget = np.dot(InputOfEnhanceLayerWithBiasTarget, self.d_t)
      OutputOfEnhanceLayerTarget = self.tansig(tempOfOutputOfEnhanceLayerTarget * self.e_t)
      InputOfOutputLayerTarget = np.hstack([OutputOfFeatureMappingLayerTarget, OutputOfEnhanceLayerTarget])
      OutputOfTarget = np.dot(InputOfOutputLayerTarget, self.f_t)

    return OutputOfTarget
