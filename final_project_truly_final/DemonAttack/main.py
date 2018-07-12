import two_games
import gym
import winner

first_name="DemonAttack-ram-v0"

#We initilize the first game without any checkpoint
first_game=two_games.learn_game(first_name,False)

first_game.train_network(None)

generations=100


for i in range(int(generations)):


	
	first_game.train_network("checkpoint")



winner.run(first_name)
