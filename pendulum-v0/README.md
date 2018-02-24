# Pendulum-v0 #

https://gym.openai.com/envs/Pendulum-v0/

## Description ## 

The inverted pendulum swingup problem is a classic problem in the control literature. In this version of the problem, the pendulum starts in a random position, and the goal is to swing it up so it stays upright..

## Notes ##

The challenge here is that the action space is continuous in that you can apply torque from -2.0 to +2.0.

I'll initially try this by discretizing the action space over 10 actions and we'll see where this gets us.
