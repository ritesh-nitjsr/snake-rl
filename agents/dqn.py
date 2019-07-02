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
        record_obj = np.array([state, action, reward, next_state, done])
        self.memory_store.append(record_obj)
        if( len(self.memory_store) >  self.memory_size ):
            self.memory_store.popleft()
        '''
        record_obj = {'state' : state, 'action' : action, 'reward' : reward, 'next_state' : next_state, 'done' : done}
        self.memory_store.append(record_obj)
        if( len(self.memory_store) >  self.memory_size ):
            self.memory_store.popleft()
        '''

    def get_batch(self, batch_size):
        batch_size = min(batch_size, len(self.memory_store))
        batch = random.sample(self.memory_store, batch_size)
        return np.array(batch), batch_size



class DeepQNetworkAgent(object):
    def __init__(self, env, interface, train, model, batch_size, num_episodes, memory_size, num_frames, num_checkpoints):
        self.env = env
        self.num_episodes = num_episodes
        self.memory_size = memory_size
        self.num_frames = num_frames
        self.num_checkpoints = num_checkpoints
        self.batch_size = batch_size
        self.frames = None
        self.interface = interface

        if(train):
            self.model = get_model(self.env, self.num_frames)
            self.memory = Memory(self.memory_size, self.num_frames)

        else:
            self.model = load_model(model)

    def get_state(self, obs):
        if(self.frames is None):
            self.frames = [obs] * self.num_frames
        else:
            self.frames = np.append(self.frames[1:], np.expand_dims(obs,0), axis = 0)
        return np.expand_dims(self.frames, 0)

    def train(self, discount_factor = 0.9, eps_start = 1, eps_min = 0.1, exploration_phase_size = 0.5):
        eps = eps_start
        eps_decay = (eps_start - eps_min)/(self.num_episodes * exploration_phase_size)
        for episode in range(1, self.num_episodes+1):
            self.frames = None
            obs = self.env.reset()
            #self.frames = [observation] * self.num_frames
            state = self.get_state(obs)
            #state = np.expand_dims(frames,0)
            loss = 0.0
            fruits_eaten = 0
            timesteps_suvived = 0
            total_reward = 0
            t = 0
            action_counts = np.zeros(self.env.num_actions)

            while(1):
                if(self.interface == 'gui'):
                    self.env.render()
                eps_i = np.random.random(1)[0]

                if(eps_i < eps):
                    action = np.random.randint(self.env.num_actions)
                else:
                    prediction = self.model.predict(state)
                    #print(prediction)
                    action = np.argmax(prediction[0])

                action_counts[action] = action_counts[action] + 1
                temp = np.array(self.frames)
                obs, reward, done, info = self.env.step(action)
                self.frames = np.array(temp)
                total_reward = total_reward + reward
                next_state = self.get_state(obs)
                self.memory.add_record(state, action, reward, next_state, done)
                state = next_state

                batch, batch_size = self.memory.get_batch(self.batch_size)

                batch_states = np.stack(batch[:,0], axis=0).reshape((batch_size, self.num_frames, self.env.height, self.env.width))
                batch_actions = batch[:,1]
                batch_rewards = batch[:,2]
                batch_next_states = np.stack(batch[:,3], axis=0).reshape((batch_size, self.num_frames, self.env.height, self.env.width))
                batch_done = batch[:,4]

                inputs = batch_states
                targets = self.model.predict(batch_states)
                Q_next_states = self.model.predict(batch_next_states)
                targets[np.arange(batch_size), list(batch_actions)] = batch_rewards + (1 - batch_done) * discount_factor * np.max(Q_next_states, axis=1)
                
                '''
                inputs = np.zeros((batch_size,  self.num_frames, self.env.height, self.env.width))
                targets = np.zeros((batch_size, self.env.num_actions))

                for i,record in enumerate(batch):
                    inputs[i:i+1] = record['state']
                    state_t = record['state']
                    action_t = record['action']
                    reward_t = record['reward']
                    next_state_t = record['next_state']
                    done_t = record['done']

                    targets[i] = self.model.predict(state_t)

                    if(done):
                        targets[i, action_t] = reward_t
                    else:
                        Q_s_dash = self.model.predict(next_state_t)
                        targets[i, action_t] = reward_t + discount_factor * np.max(Q_s_dash)
                '''

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
                self.model.save('./saved_models/temp_dqn/dqn-{:06d}.model'.format(episode))
            print("Episode : {:6d} || Epsilon : {:2.2f} || Average Loss : {:8.2f} ||  Fruits eaten : {:3d} || Timesteps survived : {:4d} || Total reward : {:5d} || Action counts : {}".format(episode, eps, loss/t, fruits_eaten, timesteps_survived, total_reward, action_counts))
            if(eps > eps_min):
                eps = eps - eps_decay
        self.model.save('./saved_models/dqn-final.model')

    def take_action(self, obs):
        state = self.get_state(obs)
        action = np.argmax(self.model.predict(state))
        return action
