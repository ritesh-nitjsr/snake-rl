#!/usr/bin/env python3
import gym
import gym_snake_game
import argparse
from agents import DeepQNetworkAgent

DQN_MODEL_PATH = None
EPISODES = 10000
NUM_FRAMES = 4
MEMORY_SIZE = 1000
NUM_CHECKPOINTS = 10
BATCH_SIZE = 64



def train(args):
    if(args.model == 'dqn'):
        env = gym.make('snake-game-v0', interface = args.interface)

        model = DeepQNetworkAgent(env = env, interface = args.interface, train = True, model = DQN_MODEL_PATH, batch_size = BATCH_SIZE,
        num_episodes = EPISODES, memory_size = MEMORY_SIZE, num_frames = NUM_FRAMES, num_checkpoints = NUM_CHECKPOINTS)

        model.train()




def play(args):
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
