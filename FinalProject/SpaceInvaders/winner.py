import argparse
import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel
from scipy.spatial import distance
from math import sqrt


### User Params ###

# The name of the game to solve
game_name = "SpaceInvaders-ram-v0"

### End User Params ###



def simulate_species(net, env, episodes=1, steps=5000, render=False):
    fitnesses = []
    price=0.2
    for runs in range(episodes):

        inputs= my_env.reset()
        #print("OBS ",obs)
        
        #print("Inputs ",inputs)
        #print("Inputs",inputs)
        cum_reward = 0.0
        for j in range(steps):
            outputs = net.serial_activate(inputs)
            action = np.argmax(outputs)
            #   print("Outputs, ",outputs)
            inputs, reward, done, _ = env.step(action)
            
            #print("Inputs ",inputs)
            #print(inputs)
            if render:
                env.render()
            if done:
                break
            cum_reward += reward

        fitnesses.append(cum_reward)

    fitness = (np.array(fitnesses).mean())
    print("Species fitness: %s" % str(fitness))
    return fitness


my_env = gym.make(game_name)


inp=input("File of the winner in format file.pkl")
import pickle
file = open(inp, 'rb')
winner=pickle.load(file)
file.close()

winner_net = nn.create_feed_forward_phenotype(winner)
for i in range(100):
    simulate_species(winner_net,my_env, 1, 1000, render=True)



train_network(my_env)
