
import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel
import pickle
import csv
from scipy import stats


#winner=input("File name of the winner pickle file ")
#game_name=input("File name of the game to test ")



winner_1="winner_DemonAttack.pkl"
winner_2="winner_both.pkl"

game_name="DemonAttack-ram-v0"
my_env=gym.make(game_name)

scores_one=[]
scores_two=[]


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
				if reward>0:
					reward=1	
				
				cum_reward += reward

			fitnesses.append(cum_reward)

		fitness = (np.array(fitnesses).mean())
		print("Species fitness: %s" % str(fitness))
		scores.append(fitness)
		return fitness





with open(winner_1,'rb') as pickle_file:
	winner_a=pickle.load(pickle_file)

with open(winner_2,'rb') as pickle_file:
	winner_b=pickle.load(pickle_file)





winner_one = nn.create_feed_forward_phenotype(winner_a)
winner_two = nn.create_feed_forward_phenotype(winner_b)

for i in range(100):
	simulate_species(scores_one,winner_one,my_env, 1, 50000, False)
	simulate_species(scores_two,winner_two,my_env, 1, 50000, False)


t_test=stats.ttest_rel(scores_one,scores_two)


with open('t-test.txt', 'a') as the_file:
	the_file.write('Paired t-test for {0} and {1} = {2} . Total scores for {0} is :{3} . Total scores for {1} is :{4}'.format(winner_1,winner_2,str(t_test),str(np.sum(scores_one)),str(np.sum(scores_two))))







