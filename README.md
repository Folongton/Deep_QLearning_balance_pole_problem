# DQL_balance_pole_visual_studio

My attempt on solving balancing pole problem via DQL implementation. <br/>
Later on plannig to switch to space invadors and eventually to Counter Strike games.

*To save time navigating, please refer to the below contents and list of tools.*

####Files:

main.py <br/>
classes.py <br/>

### Structure of solution and tools used:

##### Python packages used : 
1. Common python modules: math, random, itertools.count()
2. Traning enviroment: gym
3. Working with tensors: numpy, tensorflow
4. Models: keras.Model
4. Moving avg: pandas
4. Plotting and Image preprocessing: matplotlib, Image from PIL(python imaging Library)

#### Structure of solution:

1. Initializing replay memory.
2. Initializing 1st (Policy) Network.
3. Copying the Policy Network into 2nd (Target) Network.
   *Episode loop:*
      1. Initializing (reseting) starting state.
      *Time Step Loop:*
          1. Selecting action using exploration/exploitation condition based on epsilon function and current step.
          2. Executing action in gym enviroment.
          3. Observing reward and next state.
          4. Adding experience ( state, action, reward, next state) to the replay memory.
          5. Sampling random batch from replay memory.
          6. 
