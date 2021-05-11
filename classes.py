import gym

import math
import random
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import namedtuple
import tensorflow as tf

from PIL import Image

class DQN():
    def __init__(self, img_height, img_width):
        self.height = img_height
        self.width = img_width
        
    
    def compile_model(self, learning_rate):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model =  tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.height, self.width, 3)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')
    ])
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        return model

class CartPoleEnvManager():
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
        
    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):        
        _, reward, self.done, _ = self.env.step(action)
        return tf.convert_to_tensor([reward])
    
    def just_starting(self):
        return self.current_screen is None
    
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = tf.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
        
    def get_screen_height(self):
        screen = self.get_processed_screen()
        height = screen.shape[1]
        return height

    def get_screen_width(self):
        screen = self.get_processed_screen()
        width = screen.shape[2]
        return width
    
    def get_processed_screen(self):
        screen = self.render('rgb_array')
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    def crop_screen(self, screen):
        screen_height = screen.shape[0]
        screen_width = screen.shape[1]
        
        # Strip off top and bottom, right and left
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        left = int(screen_width * 0.25)
        right = int(screen_width * 0.75)
        screen = screen[top:bottom, left:right , :]
        
        return screen
    
    def transform_screen_data(self, screen):       
        # Convert to float, rescale, convert to tensor
        screen_small = tf.image.resize(screen, [40,90])
        screen_a = np.ascontiguousarray(screen_small, dtype=np.float32) / 255
        screen_t = tf.convert_to_tensor(screen_a)
        
        # add a batch dimension (BCHW) if needed later : tf.expand_dims(screen_t, 0)
        return tf.expand_dims(screen_t, 0)

class Plotting():

    def get_moving_average(period, values): 
        values = pd.Series(values)
        ma = values.rolling(period).mean()
        ma = ma.fillna(0)
        ma = ma.to_numpy()
        return ma

    def plot(values, moving_avg_period): 
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(values)

        moving_avg = Plotting.get_moving_average(moving_avg_period, values)
        plt.plot(moving_avg)
        print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
        plt.show(block=False)

class Agent():
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions) # explore      
        else:
                output = policy_net.predict(state) # exploit 
                print(output)
                best_action = tf.math.argmax(output, axis=1)
                return best_action.numpy().item()

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class QValues():
    @staticmethod
    def get_current(policy_net, states, actions):
        output = policy_net.predict(states)
        print(output)
        return output

    @staticmethod        
    def get_next(target_net, next_states):   
        flattened = tf.reshape(next_states, [256, (next_states.shape[1] * next_states.shape[2] * next_states.shape[3])])
        max_value = tf.math.argmax(flattened, axis=1)
        print(max_value)
        final_state_locations = tf.stack([flattened[0][x] == 0 for x in max_value.numpy()])

        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]

        values = tf.zeros_like(non_final_state_locations, dtype=tf.float32)
        next_q_values = target_net.predict(non_final_states)
        max_q_values = np.amax(next_q_values, axis=1)
        indexes = tf.where(non_final_state_locations)
        values[non_final_state_locations] = max_q_values
        return values

Experience = namedtuple( 'Experience', ('state', 'action', 'next_state', 'reward') )
   



