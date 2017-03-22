# Implementing Double DQN

Implementation of [Double Q-learning](https://arxiv.org/abs/1509.06461) in TensorFlow.

Double Q-learning was proposed as a way for alleviating the problem of overestimating the action values. It showed to perform significantly better in many atari games in comparison to the standard DQN implementation.

Implementing Double Q is actually quite simple once we have the standard DQN. I will use the previous DQN code and I will highlight the required modifications.
