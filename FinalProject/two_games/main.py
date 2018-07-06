import two_games
import gym
import visualize
import winner

first_name="SpaceInvaders-ram-v0"
second_name="DemonAttack-ram-v0"
#We initilize the first game without any checkpoint
first_game=two_games.learn_game(first_name,False)
second_game=two_games.learn_game(second_name,False)
first_game.train_network(None)

generations=50/2

#The we make 100 generations (2 generations each iteration), each generation the programm will open the checkpoint
#from the first game and use the same network on the second game

for i in range(generations):


	second_game.train_network("checkpoint")
	first_game.train_network("checkpoint")



winner.run(first_name,second_name)
