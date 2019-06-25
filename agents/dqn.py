from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.models import load_model
import numpy as np
from collections import deque
import random
import time

def get_model(env, num_frames):
    input = Input(shape = (num_frames, env.height, env.width))
    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_first')(input)
    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_first')(X)
    X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    X = Dense(env.num_actions, activation='sigmoid')(X)
    model = Model(inputs = input, outputs = X, name = 'CNN')
    model.summary()
    model.compile(optimizer=RMSprop(), loss='mse')
    return model

class Memory(object):
    def __init__(self, memory_size, num_frames):
        self.memory_size = memory_size
        self.memory_store = deque()
        self.num_frames = num_frames

    def reset(self):
        self.memory_store = deque()

    def add_record(self, state, action, reward, next_state, done):
        record_obj = {'state' : state, 'action' : action, 'reward' : reward, 'next_state' : next_state, 'done' : done}
        self.memory_store.append(record_obj)
        if( len(self.memory_store) >  self.memory_size ):
            self.memory_store.popleft()

    def get_batch(self, batch_size):
        batch_size = min(batch_size, len(self.memory_store))
        batch = random.sample(self.memory_store, batch_size)
        return batch, batch_size


class DeepQNetworkAgent(object):
    def __init__(self, env, interface, train, model, batch_size, num_episodes, memory_size, num_frames, num_checkpoints):
        self.env = env
        self.num_episodes = num_episodes
        self.memory_size = memory_size
        self.num_frames = num_frames
        self.num_checkpoints = num_checkpoints
        self.batch_size = batch_size
        self.frames = deque()
        self.interface = interface

        if(train):
            self.model = get_model(self.env, self.num_frames)
            self.memory = Memory(self.memory_size, self.num_frames)
            self.train()
        else:
            self.model = load_model(model)

    def insert_last_frames(self, observation):
        if(len(self.frames) == self.num_frames):
            self.frames.popleft()
        self.frames.append(observation)


    def get_state(self):
        state = []
        for frame in self.frames:
            state.append(frame)
        if(len(state) < self.num_frames):
            last_frame = state[-1]
            for _ in range(self.num_frames - len(state)):
                state.append(last_frame)
        state = np.array(state)
        return np.expand_dims(state, axis=0)

    def train(self, discount_factor = 0.95, eps_start = 1, eps_min = 0.05, eps_decay = 0.9999):
        eps = eps_start
        for episode in range(1, self.num_episodes+1):
            obs = self.env.reset()
            self.insert_last_frames(obs)
            state = self.get_state()
            loss = 0.0
            eps = max(eps*eps_decay,eps_min)
            fruits_eaten = 0
            timesteps_suvived = 0
            total_reward = 0

            t = 0

            while(1):
                if(self.interface == 'gui'):
                    self.env.render()
                eps_i = np.random.random(1)[0]

                if(eps_i < eps):
                    action = np.random.randint(self.env.num_actions)
                else:
                    action = np.argmax((self.model.predict(state))[0])

                obs, reward, done, info = self.env.step(action)
                total_reward = total_reward + reward
                self.insert_last_frames(obs)
                next_state =  self.get_state()
                self.memory.add_record(state, action, reward, next_state, done)

                batch, batch_size = self.memory.get_batch(self.batch_size)

                inputs = np.zeros((batch_size,  self.num_frames, self.env.height, self.env.width))
                targets = np.zeros((batch_size, self.env.num_actions))

                for i,record in enumerate(batch):
                    inputs[i] = record['state']
                    state_t = record['state']
                    action_t = record['action']
                    reward_t = record['reward']
                    next_state_t = record['next_state']
                    done_t = record['done']

                    targets[i] = self.model.predict(state_t)

                    if(done):
                        targets[i, action_t] = reward_t
                    else:
                        Q_s_dash = np.zeros(self.env.num_actions)
                        targets[i, action_t] = reward_t + discount_factor * np.max(Q_s_dash)

                loss = loss + self.model.train_on_batch(inputs, targets)
                t = t + 1



                if(done or t>=1000):
                    if(done):
                        fruits_eaten = info['Total Fruits eaten']
                        timesteps_survived = info['Total timesteps suvived']
                    else:
                        fruits_eaten = -1
                        timesteps_survived = 1000
                    break
            if((episode % (self.num_episodes/self.num_checkpoints)) == 0):
                self.model.save('./saved_models/temp_dqn/dqn-{:08d}.model'.format(episode))
            print("Episode : {} || Loss : {} ||  Fruits eaten : {} || Timesteps survived : {} || Total reward : {} ".format(episode, loss, fruits_eaten, timesteps_survived, total_reward))
        self.model.save('./saved_models/model/temp_dqn/dqn-final.model')

    def take_action(self):
        pass
