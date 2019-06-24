from gym.envs.registration import register

register(
    id='snake-game-v0',
    entry_point='gym_snake_game.envs:SnakeGameEnv',
)
