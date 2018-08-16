
import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel
import pickle



def run(first_game,second_game):


    def simulate_species(net, env, episodes, steps, render):
        fitnesses = []
      
        for runs in range(episodes):

            inputs= my_env.reset()
           
            cum_reward = 0.0
            for j in range(steps):
                outputs = net.serial_activate(inputs)
                action = np.argmax(outputs)
                
                inputs, reward, done, _ = env.step(action)
                
                if render:
                    env.render()
                if done:
                    break
                cum_reward += reward

            fitnesses.append(cum_reward)

        fitness = (np.array(fitnesses).mean())
        print("Species fitness: %s" % str(fitness))
        return fitness


    to_choose=("Please enter 1 if you want to test ",first_game," or 2 if you want to test ",second_game)

    choice=input(to_choose)

    if choice==1:

        my_env = gym.make(first_game)

    else:

        my_env = gym.make(second_game)

    file = open('winner.pkl', 'rb')
    winner=pickle.load(file)
    file.close()




    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net,my_env, 1, 8000, True)



    train_network(my_env)
