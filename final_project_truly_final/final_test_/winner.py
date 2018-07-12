
import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel
import pickle
import csv


#winner=input("File name of the winner pickle file ")
#game_name=input("File name of the game to test ")


winner="winner_both.pkl"
game_name="SpaceInvaders-ram-v0"
my_env=gym.make(game_name)

scores=[]

def simulate_species(scores,net, env, episodes, steps, render):
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
		scores.append(fitness)
		return fitness





with open("winner_both.pkl",'rb') as pickle_file:
	winner_a=pickle.load(pickle_file)





winner_net = nn.create_feed_forward_phenotype(winner_a)
for i in range(100):
	simulate_species(scores,winner_net,my_env, 1, 50000, False)


final_score=(np.mean(scores)-np.min(scores))/(np.max(scores)-np.min(scores))

mx=np.max(scores)
with open('one.txt', 'a') as the_file:
    the_file.write('Mean scores for game {0} and trained on {1}: {2} with MAX={3}'.format(game_name,winner, final_score,mx))








