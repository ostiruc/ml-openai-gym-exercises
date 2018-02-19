# CartPole-v0 #

https://gym.openai.com/envs/CartPole-v0/

## Description ## 

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The system is controlled by applying a force of +1 or -1 to
the cart. The pendulum starts upright, and the goal is to prevent it from
falling over. A reward of +1 is provided for every timestep that the pole
remains upright. The episode ends when the pole is more than 15 degrees from
vertical, or the cart moves more than 2.4 units from the center.

## Notes ##

I'm attempting to get this working using base Tensorflow and Deep Q-Learning.
I'm going to be following the methods laid out here:

https://keon.io/deep-q-learning/
