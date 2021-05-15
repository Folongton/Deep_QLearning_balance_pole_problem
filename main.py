import numpy as np

from itertools import count

import tensorflow as tf
from classes import Experience, DQN, CartPoleEnvManager, Plotting, Agent, EpsilonGreedyStrategy, ReplayMemory, QValues

def check_unprocessed_screen():
    em = CartPoleEnvManager()
    em.reset()
    screen = em.render('rgb_array')

    plt.figure()
    plt.imshow(screen)
    plt.title('Non-processed screen example')
    plt.show(block=True)
def check_processed_screen():
    em = CartPoleEnvManager()
    em.reset()
    screen = em.get_processed_screen()
    screen = tf.squeeze(screen, 0)

    figure(figsize=(12, 5), dpi=80)
    plt.imshow(screen)
    plt.title('Processed screen example')
    plt.show(block=True)
def check_starting_screen():
    em = CartPoleEnvManager()
    screen = em.get_state()
    screen = tf.squeeze(screen, 0)

    plt.figure(figsize=(12, 5), dpi=80)
    plt.imshow(screen)
    plt.title('Starting state example')
    plt.show(block=True)
def check_non_starting_state():
    em = CartPoleEnvManager()
    for i in range(5):
        em.take_action(np.random.randint(2))
        screen = em.get_state()
        screen = tf.squeeze(screen, 0)
       
        plt.figure(figsize=(12, 5), dpi=80)
        plt.imshow(screen, interpolation='none')
        plt.title('Non starting state example')
  
        plt.show(block=True)
def check_moving_avg_plot():

    Plotting.plot(np.random.rand(300),100)

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = tf.squeeze(tf.stack(batch.state), axis=1)
    t2 = tf.convert_to_tensor(batch.action)
    t3 = tf.squeeze(tf.stack(batch.reward), axis=1)
    t4 = tf.squeeze(tf.stack(batch.next_state), axis=1)

    return (t1,t2,t3,t4)


#check_moving_avg_plot()
# Hyperparameters
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100_000
lr = 0.001
num_episodes = 1200


# Setting all objects
em = CartPoleEnvManager()
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

agent = Agent(strategy, em.num_actions_available())
memory = ReplayMemory(memory_size)


policy_net = DQN(em.get_screen_height(), em.get_screen_width())
policy_net = policy_net.def_model()

target_net = DQN(em.get_screen_height(), em.get_screen_width())
target_net = target_net.def_model()
target_net.set_weights(policy_net.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

episode_durations = []


for episode in range(num_episodes):

    em.reset()
    state = em.get_state()

    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            #print("These are rewards for this batch: " + str(rewards.numpy()))


            for var in optimizer.variables():
                var.assign(tf.zeros_like(var)) # reseting optimizer weights

            with tf.GradientTape() as tape:
                tape.watch(states)
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards
                loss = tf.keras.losses.MSE(current_q_values, target_q_values)
                print('Loss for this batch is : ', loss.numpy())
                gradients = tape.gradient(loss, policy_net.trainable_variables)
                optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables)) # updating the weights in the model


        if em.done:
            episode_durations.append(timestep)
            Plotting.plot(episode_durations, 100)
            break

        if episode % target_update == 0:
            target_net.set_weights(policy_net.get_weights())