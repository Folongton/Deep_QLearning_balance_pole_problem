# DQL_balance_pole_visual_studio

My attempt on solving balancing pole problem via Deep Q-Learning implementation. <br/>
Later on plannig to switch to space invadors and eventually to Counter Strike games.

*To save time navigating, please refer to the below contents and list of tools.*

#### Files:

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

1. Initialize replay memory.
2. Initialize 1st (Policy) Network.
3. Copy the Policy Network into 2nd (Target) Network. <br/>
   *Episode loop:*
      1. Initialize (reset) starting state. <br/>
      *Time Step Loop:*
          1. Select action using exploration/exploitation condition based on epsilon function and current step.
          2. Execute action in gym enviroment.
          3. Observe reward and next state.
          4. Add experience ( state, action, reward, next state) to the replay memory.
          5. Sample random batch from replay memory.
          6. Preprocess batch of experiences.
          7. Forward feed batch of experiences to 1st (Policy) Network.
          8. Forward feed output values (Q values) from 1st Network to 2nd (Target) Network.
          9. Calculate loss between Q values from 1st Network and Q values from 2nd Network.
          10. Update weights in the 1st (policy) Network via backpropagation to minimize loss.
          11. After x steps (10 in our case), weights in 2nd (Target) Network are updated to the weights of 1st (policy) Network.
4. Continue iteration until ANN approximate Q-function to the optimal state e.g. solving the moving pole problem. 
         
*Although this solution does not converge the function to the point of optimal Q-fuction yet. (I think, I have made some mistake in data flow). I have learned a lot from it: Q-Learning, ANNs inner structure and process of backpropagation, General concepts of Reinforced learning, tensor manipulation and image preprocessing. I will definitelly come back to finish this one in a few weeks, with a clearer head and new knowledge.*
