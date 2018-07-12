import argparse
import gym
import os
import numpy as np
from neat import nn, population, statistics, parallel
from scipy.spatial import distance
from math import sqrt
import pickle



class learn_game:


	
	max_steps=10000
	episodes=1
	render=False
	generations=1
	checkpoint=None
	num_cores=1

	def __init__(self,game_name,render):

		self.game_name=game_name
		self.my_env=gym.make(game_name)
		
	

	def simulate_species(self,net,env,episodes, steps, render):
		fitnesses = []
		
		for runs in range(episodes):
			inputs = env.reset()
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

		fitness = np.array(fitnesses).mean()
		print("Species fitness: %s" % str(fitness))
		return fitness


	
	

	
	def worker_evaluate_genome(self,g):
		net = nn.create_feed_forward_phenotype(g)
		return self.simulate_species(net, self.my_env, self.episodes, self.max_steps, self.render)


	def evaluate_genome(self,g):
		net = nn.create_feed_forward_phenotype(g)
		return self.simulate_species(net,self.my_env, self.episodes, self.max_steps, self.render)

	def eval_fitness(self,genomes):
		for g in genomes:
			fitness = self.evaluate_genome(g)
			g.fitness = fitness

	def train_network(self,checkpoint):

		self.checkpoint=checkpoint
		# Simulation

		local_dir = os.path.dirname(__file__)
		config_path = os.path.join(local_dir, 'config.txt')
		pop = population.Population(config_path)

		# Load checkpoint
		if self.checkpoint!= None:
			pop.load_checkpoint(self.checkpoint)
		
		#pe = parallel.ParallelEvaluator(args.numCores,worker_evaluate_genome)
		pop.run(self.eval_fitness, self.generations)

		pop.save_checkpoint("checkpoint")

		# Log statistics.
		statistics.save_stats(pop.statistics)
		statistics.save_species_count(pop.statistics)
		statistics.save_species_fitness(pop.statistics)

		print('Number of evaluations: {0}'.format(pop.total_evaluations))

		# Show output of the most fit genome against training data.
		winner = pop.statistics.best_genome()

		# Save best network
		import pickle
		with open('winner.pkl', 'wb') as output:
			pickle.dump(winner, output, 1)

	
	
