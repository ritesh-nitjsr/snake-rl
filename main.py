#!/usr/bin/env python3
import gym
import gym_snake_game
import argparse
from agents import DeepQNetworkAgent

DQN_MODEL_PATH = 'dqn-final.model'
TRAIN_EPISODES = 60000
TEST_EPISODES = 100
NUM_FRAMES = 4
MEMORY_SIZE = 1000
NUM_CHECKPOINTS = 10
BATCH_SIZE = 50



def train(args):
    if(args.model == 'dqn'):
        env = gym.make('snake-game-v0', interface = args.interface)

        model = DeepQNetworkAgent(env = env, interface = args.interface, train = True, model = DQN_MODEL_PATH, batch_size = BATCH_SIZE,
        num_episodes = TRAIN_EPISODES, memory_size = MEMORY_SIZE, num_frames = NUM_FRAMES, num_checkpoints = NUM_CHECKPOINTS)

        model.train()




def play(args):
    if(args.model == 'dqn'):
        env = gym.make('snake-game-v0', interface = args.interface)
        model = DeepQNetworkAgent(env = env, interface = args.interface, train = False, model = DQN_MODEL_PATH, batch_size = None,
        num_episodes = None, memory_size = None, num_frames = NUM_FRAMES, num_checkpoints = None)

        for episode in range(TEST_EPISODES):
            obs = env.reset()
            total_reward = 0
            fruits_eaten = 0
            timesteps_survived = 0
            t = 0
            while(1):
                if(args.interface == 'gui'):
                    env.render()
                action = model.take_action(obs)
                obs, reward, done, info = env.step(action)
                total_reward = total_reward + reward
                t = t + 1
                if(done or t>=1000):
                    if(not done):
                        fruits_eaten = -1
                        timesteps_survived = 1000
                    else:
                        fruits_eaten = info['Total Fruits eaten']
                        timesteps_survived = info['Total timesteps suvived']
                    break
            print("Episode : {} || Fruits eaten : {} || Timesteps survived : {} || Total reward : {} ".format(episode, fruits_eaten, timesteps_survived, total_reward))



    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices = ['play', 'train'], default = 'train', help = 'play or train mode')
    parser.add_argument("--interface", type=str, choices = ['gui', 'cli'], default = 'cli', help = 'gui or cli interface')
    parser.add_argument("--model", type=str, choices = ['dqn'], default = 'dqn', help = 'model to be used for training or play')

    args = parser.parse_args()

    if(args.mode == 'play'):
        play(args)

    elif(args.mode == 'train'):
        train(args)



if __name__ == "__main__":
    main()
