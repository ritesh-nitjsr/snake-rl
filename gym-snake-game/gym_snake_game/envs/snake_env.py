import gym
from gym import error, spaces, utils
from collections import deque
import random
import time
import numpy as np
try:
    from gym.envs.classic_control import rendering
except:
    pass

class CellState(object):
    EMPTY = 0
    WALL = 1
    FRUIT = 2
    SNAKE_HEAD = 3
    SNAKE_BODY = 4

class SnakeAction(object):
    MAINTAIN_DIRECTION = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2

class Rewards(object):
    EATEN_FRUIT = 100
    ALIVE = 0
    DEAD = -50

class Directions(object):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class Colors(object):
    SNAKE_BODY = (0,200,0)
    WALL = (0,0,0)
    FRUIT = (255,0,0)
    SNAKE_HEAD = (0,0,150)
    EMPTY = (255,255,255)

class SnakeGame(object):
    def __init__(self, height, width, init_pos, init_length=3):
        self.height = height
        self.width = width
        self.length = init_length

        self.state = np.empty((self.height,self.width))

        self.snake = deque()

        for x in range(self.height):
            for y in range(self.width):
                if(x==0 or y==0 or x==self.height-1 or y==self.width-1):
                    self.state[x,y] = CellState.WALL
                else:
                    self.state[x,y] = CellState.EMPTY

        self.state[init_pos] = CellState.SNAKE_HEAD

        self.current_head = None

        self.add_head(init_pos)

        for i in range(init_length-1):
             self.add_tail((init_pos[0]+i+1, init_pos[1]))

        self.current_direction = Directions.NORTH

        self.generate_fruit()

    def generate_fruit(self):
        '''
        if(self.current_head[0]+1 < self.height-1 and self.state[self.current_head[0]+1, self.current_head[1]]==CellState.EMPTY):
            self.state[self.current_head[0]+1, self.current_head[1]] = CellState.FRUIT
            return

        if(self.current_head[0]-1 > 0 and self.state[self.current_head[0]-1, self.current_head[1]]==CellState.EMPTY):
            self.state[self.current_head[0]-1, self.current_head[1]] = CellState.FRUIT
            return

        if(self.current_head[1]+1 < self.width-1 and self.state[self.current_head[0], self.current_head[1]+1]==CellState.EMPTY):
            self.state[self.current_head[0], self.current_head[1]+1] = CellState.FRUIT
            return

        if(self.current_head[1]-1 > 0 and self.state[self.current_head[0], self.current_head[1]-1]==CellState.EMPTY):
            self.state[self.current_head[0], self.current_head[1]-1] = CellState.FRUIT
            return
        '''

        while(1):
            x = np.random.randint(low=1,high=self.height,size=1)[0]
            y = np.random.randint(low=1,high=self.width,size=1)[0]
            #print(x,y, self.state[x,y])
            if(self.state[x,y] == CellState.EMPTY):
                self.state[x,y] = CellState.FRUIT
                break



    def add_head(self,pos):
        self.snake.appendleft(pos)
        self.state[pos] = CellState.SNAKE_HEAD
        if(self.current_head != None):
            self.state[self.current_head] = CellState.SNAKE_BODY
        self.current_head = pos

    def add_tail(self,pos):
        self.snake.append(pos)
        self.state[pos] = CellState.SNAKE_BODY

    def get_next_head(self, direction):
        next_head = list(self.current_head)
        if(direction == Directions.NORTH):
            next_head[0] = next_head[0] - 1
        elif(direction == Directions.SOUTH):
            next_head[0] = next_head[0] + 1
        elif(direction == Directions.EAST):
            next_head[1] = next_head[1] + 1
        elif(direction == Directions.WEST):
            next_head[1] = next_head[1] - 1
        return tuple(next_head)

    def remove_tail(self):
        pos = self.snake.pop()
        self.state[pos] = CellState.EMPTY



class SnakeGameEnv(gym.Env):
    metadata= {'render.modes': ['human']}

    def __init__(self, interface='cli'):
        self.width = 10
        self.height = 10
        self.init_pos = (5,5)
        self.num_actions = 3
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = 0, high = 4,  shape = (self.height, self.width))
        self.snake_game = SnakeGame(self.height, self.width, self.init_pos)
        self.viewer = None
        self.fruits_eaten = 0
        self.timesteps_suvived = 0
        self.sleep_time = 0
        if(interface=='gui'):
            self.sleep_time = 0.0

    def step(self, action):
        time.sleep(self.sleep_time)
        observation = None
        reward = 0
        done = 0
        info = None

        if(action == SnakeAction.MAINTAIN_DIRECTION):
            next_head = self.snake_game.get_next_head(self.snake_game.current_direction)

            if(self.snake_game.state[next_head] == CellState.EMPTY):
                self.snake_game.add_head(next_head)
                self.snake_game.remove_tail()
                reward = Rewards.ALIVE
            elif(self.snake_game.state[next_head] == CellState.FRUIT):
                self.snake_game.add_head(next_head)
                reward = Rewards.EATEN_FRUIT + Rewards.ALIVE
                self.fruits_eaten = self.fruits_eaten + 1
                self.snake_game.generate_fruit()
            else:
                self.snake_game.add_head(next_head)
                self.snake_game.remove_tail()
                done = 1
                reward = Rewards.DEAD

        elif(action == SnakeAction.TURN_LEFT):
            next_head = None

            if(self.snake_game.current_direction == Directions.NORTH):
                next_head = self.snake_game.get_next_head(Directions.WEST)
                self.snake_game.current_direction = Directions.WEST
            elif(self.snake_game.current_direction == Directions.SOUTH):
                next_head = self.snake_game.get_next_head(Directions.EAST)
                self.snake_game.current_direction = Directions.EAST
            elif(self.snake_game.current_direction == Directions.WEST):
                next_head = self.snake_game.get_next_head(Directions.SOUTH)
                self.snake_game.current_direction = Directions.SOUTH
            else:
                next_head = self.snake_game.get_next_head(Directions.NORTH)
                self.snake_game.current_direction = Directions.NORTH


            if(self.snake_game.state[next_head] == CellState.EMPTY):
                self.snake_game.add_head(next_head)
                self.snake_game.remove_tail()
                reward = Rewards.ALIVE
            elif(self.snake_game.state[next_head] == CellState.FRUIT):
                self.snake_game.add_head(next_head)
                reward = Rewards.EATEN_FRUIT + Rewards.ALIVE
                self.fruits_eaten = self.fruits_eaten + 1
                self.snake_game.generate_fruit()
            else:
                self.snake_game.add_head(next_head)
                self.snake_game.remove_tail()
                done = 1
                reward = Rewards.DEAD

        else:
            next_head = None

            if(self.snake_game.current_direction == Directions.NORTH):
                next_head = self.snake_game.get_next_head(Directions.EAST)
                self.snake_game.current_direction = Directions.EAST
            elif(self.snake_game.current_direction == Directions.SOUTH):
                next_head = self.snake_game.get_next_head(Directions.WEST)
                self.snake_game.current_direction = Directions.WEST
            elif(self.snake_game.current_direction == Directions.WEST):
                next_head = self.snake_game.get_next_head(Directions.NORTH)
                self.snake_game.current_direction = Directions.NORTH
            else:
                next_head = self.snake_game.get_next_head(Directions.SOUTH)
                self.snake_game.current_direction = Directions.SOUTH


            if(self.snake_game.state[next_head] == CellState.EMPTY):
                self.snake_game.add_head(next_head)
                self.snake_game.remove_tail()
                reward = Rewards.ALIVE
            elif(self.snake_game.state[next_head] == CellState.FRUIT):
                self.snake_game.add_head(next_head)
                reward = Rewards.EATEN_FRUIT + Rewards.ALIVE
                self.fruits_eaten = self.fruits_eaten + 1
                self.snake_game.generate_fruit()
            else:
                self.snake_game.add_head(next_head)
                self.snake_game.remove_tail()
                done = 1
                reward = Rewards.DEAD

        if(done):
            info = {'Total Fruits eaten' : self.fruits_eaten , 'Total timesteps suvived' : self.timesteps_suvived}
        else:
            self.timesteps_suvived = self.timesteps_suvived + 1

        observation = self.snake_game.state

        return observation, reward, done, info

    def reset(self):
        self.snake_game = SnakeGame(self.height, self.width, self.init_pos)
        observation = self.snake_game.state
        self.fruits_eaten = 0
        self.timesteps_suvived = 0
        return observation

    def render(self, mode='human', close=False):
        frame_height = 250
        frame_width = 250
        wsf = frame_width/self.width
        hsf = frame_height/self.height

        if self.viewer is None:
            self.viewer = rendering.Viewer(frame_width, frame_height)

        for x in range(self.width):
            for y in range(self.height):
                l, r, t, b = x*wsf, (x+1)*wsf, y*hsf, (y+1)*hsf
                square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                if(self.snake_game.state[y,x] == CellState.EMPTY):
                    square.set_color(Colors.EMPTY[0], Colors.EMPTY[1], Colors.EMPTY[2])
                if(self.snake_game.state[y,x] == CellState.FRUIT):
                    square.set_color(Colors.FRUIT[0], Colors.FRUIT[1], Colors.FRUIT[2])
                if(self.snake_game.state[y,x] == CellState.WALL):
                    square.set_color(Colors.WALL[0], Colors.WALL[1], Colors.WALL[2])
                if(self.snake_game.state[y,x] == CellState.SNAKE_HEAD):
                    square.set_color(Colors.SNAKE_HEAD[0], Colors.SNAKE_HEAD[1], Colors.SNAKE_HEAD[2])
                if(self.snake_game.state[y,x] == CellState.SNAKE_BODY):
                    square.set_color(Colors.SNAKE_BODY[0], Colors.SNAKE_BODY[1], Colors.SNAKE_BODY[2])

                self.viewer.add_onetime(square)

        for y in range(self.width):
            line = rendering.Line(start=(y,0), end=(y,self.height))
            self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode=='rgb_array')
