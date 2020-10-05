# Modified verson of Taxi Problem
#
# References:
# Frans, K 2016, felipelodur/Q-Learning-Taxi-v2: Q-Learning-Taxi-v2, Retrieved 25th September 2020,
# <https://github.com/felipelodur/Q-Learning-Taxi-v2>

from IPython.display import clear_output
import numpy as np
import random
import time
import gym

env = gym.make("Taxi-v2")  # Create environment
env.render()  # Show it

# Number of possible actions
action_size = env.action_space.n
print("Action size ", action_size)

# Number of possible states
state_size = env.observation_space.n
print("State size ", state_size)

qtable = np.zeros((state_size, action_size))
print(qtable)

episodes = 30000            # Total episodes
max_steps = 1000            # Max steps per episode
lr = 0.3                    # Learning rate
decay_fac = 0.00001         # Decay learning rate each iteration
gamma = 0.90                # Discounting rate - later rewards impact less


for episode in range(episodes):

    state = env.reset()  # Reset the environment
    done = False        # Are we done with the environment
    lr -= decay_fac     # Decaying learning rate
    step = 0

    if lr <= 0:  # Nothing more to learn?
        break

    for step in range(max_steps):

        # Randomly Choose an Action
        action = env.action_space.sample()

        # Take the action -> observe new state and reward
        new_state, reward, done, info = env.step(action)

        # Update qtable values
        if done == True:  # If last, do not count future accumulated reward
            if(step < 199 | step > 201):
                qtable[state, action] = qtable[state, action] + \
                    lr*(reward+gamma*0-qtable[state, action])
            break
        else:  # Consider accumulated reward of best decision stream
            qtable[state, action] = qtable[state, action]+lr * \
                (reward+gamma *
                 np.max(qtable[new_state, :])-qtable[state, action])

        # if done.. jump to next episode
        if done == True:
            break

        # moving states
        state = new_state

    episode += 1

    if (episode % 3000 == 0):
        print('episode = ', episode)
        print('learning rate = ', lr)
        print('-----------')

# New environment
state = env.reset()
env.render()
done = False
total_reward = 0

while(done == False):
    action = np.argmax(qtable[state, :])  # Choose best action (Q-table)
    state, reward, done, info = env.step(action)  # Take action
    total_reward += reward  # Summing rewards

    # Display it
    time.sleep(0.5)
    clear_output(wait=True)
    env.render()
    print('Episode Reward = ', total_reward)
