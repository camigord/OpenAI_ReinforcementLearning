# Original DQN using TensorFlow

Implementing a basic version of the original [DQN](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html). The code was written inspired on the contributions from [Denny Britz](https://github.com/dennybritz/reinforcement-learning/tree/master/DQN) and [Carles Gelada](https://github.com/cgel/DRL).

Although the model was trained using a GPU, the input images (84x84) are not large enough to observe a significant speed up in training time compared to using CPUs.

## Local Environment Specifications

The model was trained and tested on a machine with:
  - Intel® Core™ i7-2700K CPU @ 3.50GHz × 8
  - 16GB RAM
  - GeForce GTX TITAN Black
  - Ubuntu 14.04 LTS
  - TensorFlow r0.11
  - Python 2.7

## Performance

The model was trained for almost 7 days.

![results]()
