# MDP-value-and-policy-iteration-
A Markov Decision Process (MDP) implementation using value and policy iteration to calculate the optimal policy.

Inputs used are a 4x3 world, a 5x5 world and a 11x11 world. The inputs can be easily modified as desired.

4x3 world:

([[-0.04, -0.04, -0.04, +1],  <br/>
[-0.04, None, -0.04, -1],  <br/>
[-0.04, -0.04, -0.04, -0.04]],  <br/>
terminals=[(3, 2), (3, 1)])

5x5 world:

([[-0.04, -0.04, -0.04, -0.04, +1],  <br/>
[-0.04, -0.04, -0.04, -0.4, -1],  <br/>
[-0.04, -0.04, None, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04]],  <br/>
terminals=[(4, 4), (4, 3)])

11x11 world:

([[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, +1],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -1],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, None, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, None, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, None, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04],  <br/>
[-0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]],  <br/>
terminals=[(10, 10), (10, 9)])
