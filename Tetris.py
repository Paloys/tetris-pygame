import random
import sys
import time

import gym
from gym import spaces
import numpy as np
import pygame
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor
from tensorflow.python.keras.layers import Flatten
from keras.utils import plot_model

"""
10 x 20 grid
play_height = 2 * play_width

tetriminos:
    0 - S - green
    1 - Z - red
    2 - I - cyan
    3 - O - yellow
    4 - J - blue
    5 - L - orange
    6 - T - purple
"""
pygame.init()

pygame.font.init()

# global variables

col = 10  # 10 columns
row = 20  # 20 rows
s_width = 800  # window width
s_height = 750  # window height
play_width = 300  # play window width; 300/10 = 30 width per block
play_height = 600  # play window height; 600/20 = 30 height per block
block_size = 30  # size of block

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height - 50

filepath = '/Users/louis/PycharmProjects/pythonProject/highscore.txt'
fontpath = '/Users/louis/PycharmProjects/pythonProject/arcade.ttf'
fontpath_mario = '/Users/louis/PycharmProjects/pythonProject/mario.ttf'

# shapes formats

S = [
        [
            "00000",
            "00000",
            "00110",
            "01100",
            "00000"
        ],
        [
            "00000",
            "00100",
            "00110",
            "00010",
            "00000"
        ]
    ]
Z = [
        [
            "00000",
            "00000",
            "01100",
            "00110",
            "00000"
        ],
        [
            "00000",
            "00100",
            "01100",
            "01000",
            "00000"
        ]
    ]
I = [
        [
            "00000",
            "00100",
            "00100",
            "00100",
            "00100"
        ],
        [
            "00000",
            "11110",
            "00000",
            "00000",
            "00000"
        ]
    ]
O = [
        [
            "00000",
            "00000",
            "01100",
            "01100",
            "00000"
        ]
    ]
J = [
        [
            "00000",
            "01000",
            "01110",
            "00000",
            "00000"
        ],
        [
            "00000",
            "00110",
            "00100",
            "00100",
            "00000"
        ],
        [
            "00000",
            "00000",
            "01110",
            "00010",
            "00000"
        ],
        [
            "00000",
            "00100",
            "00100",
            "01100",
            "00000"
        ]
    ]
L = [
        [
            "00000",
            "00010",
            "01110",
            "00000",
            "00000"
        ],
        [
            "00000",
            "00100",
            "00100",
            "00110",
            "00000"
        ],
        [
            "00000",
            "00000",
            "01110",
            "01000",
            "00000"
        ],
        [
            "00000",
            "01100",
            "00100",
            "00100",
            "00000"
        ]
    ]
T = [
    [
        "00000",
        "00100",
        "01110",
        "00000",
        "00000"
    ],
    [
        "00000",
        "00100",
        "00110",
        "00100",
        "00000"
    ],
    [
        "00000",
        "00000",
        "01110",
        "00100",
        "00000"
    ],
    [
        "00000",
        "00100",
        "01100",
        "00100",
        "00000"
    ]
]

# index represents the shape
shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]


# class to represent each of the pieces


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]  # choose color from the shape_color list
        self.rotation = 0  # chooses the rotation according to index


# initialise the grid
def create_grid(locked_pos=None):
    if locked_pos is None:
        locked_pos = {}
    grid = [[(0, 0, 0) for x in range(col)] for y in range(row)]  # grid represented rgb tuples

    # locked_positions dictionary
    # (x,y):(r,g,b)
    for y in range(row):
        for x in range(col):
            if (x, y) in locked_pos:
                grid[y][x] = 1  # set grid position to color

    return grid


def grid_for_ai_with_list(locked_pos=None):
    if locked_pos is None:
        locked_pos = []
    grid = [[0 for x in range(col)] for y in range(row)]

    for y in range(row):
        for x in range(col):
            if (x, y) in locked_pos:
                grid[y][x] = 1
    return grid


def grid_for_ai(locked_pos=None):
    if locked_pos is None:
        locked_pos = {}
    grid = [[0 for x in range(col)] for y in range(row)]

    # locked_positions dictionary
    # (x,y):(r,g,b)
    for y in range(row):
        for x in range(col):
            if (x, y) in locked_pos:
                grid[y][x] = 1  # set grid position to color

    return grid


def convert_shape_format(piece):
    positions = []
    shape_format = piece.shape[piece.rotation % len(piece.shape)]  # get the desired rotated shape from piece

    '''
    e.g.
       ['.....',
        '.....',
        '..00.',
        '.00..',
        '.....']
    '''
    for i, line in enumerate(shape_format):  # i gives index; line gives string
        row = list(line)  # makes a list of char from string
        for j, column in enumerate(row):  # j gives index of char; column gives char
            if column == '1':
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)  # offset according to the input given with dot and zero

    return positions


# checks if current position of piece in grid is valid
def valid_space(piece, grid):
    # makes a 2D list of all the possible (x,y)
    accepted_pos = [[(x, y) for x in range(col) if grid[y][x] == (0, 0, 0)] for y in range(row)]
    # removes sub lists and puts (x,y) in one list; easier to search
    accepted_pos = [x for item in accepted_pos for x in item]

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
        if pos not in accepted_pos:
            if pos[1] >= 0:
                return False
    return True


# check if piece is out of board
def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


# chooses a shape randomly from shapes list
def get_shape():
    return Piece(5, 0, random.choice(shapes))


# draws text in the middle
def draw_text_middle(text, size, color, surface):
    font = pygame.font.Font(fontpath, size, bold=False, italic=True)
    label = font.render(text, False, color)

    surface.blit(label, (
        top_left_x + play_width / 2 - (label.get_width() / 2), top_left_y + play_height / 2 - (label.get_height() / 2)))


# draws the lines of the grid for the game
def draw_grid(surface):
    r = g = b = 0
    grid_color = (r, g, b)

    for i in range(row):
        # draw grey horizontal lines
        pygame.draw.line(surface, grid_color, (top_left_x, top_left_y + i * block_size),
                         (top_left_x + play_width, top_left_y + i * block_size))
        for j in range(col):
            # draw grey vertical lines
            pygame.draw.line(surface, grid_color, (top_left_x + j * block_size, top_left_y),
                             (top_left_x + j * block_size, top_left_y + play_height))


# clear a row when it is filled
def clear_rows(grid, locked):
    # need to check if row is clear then shift every other row above down one
    increment = 0
    for i in range(len(grid) - 1, -1, -1):  # start checking the grid backwards
        grid_row = grid[i]  # get the last row
        if sum(grid_row) == len(grid_row):  # if there are no empty spaces (i.e. black blocks)
            increment += 1
            # add positions to remove from locked
            index = i  # row index will be constant
            for j in range(len(grid_row)):
                try:
                    locked.remove((j, i))  # delete every locked element in the bottom row
                except ValueError:
                    continue

    # shift every row one step down
    # delete filled bottom row
    # add another empty row on the top
    # move down one step
    if increment > 0:
        # sort the locked list according to y value in (x,y) and then reverse
        # reversed because otherwise the ones on the top will overwrite the lower ones
        for key in sorted(locked, key=lambda a: a[1])[::-1]:
            x, y = key
            if y < index:  # if the y value is above the removed index
                new_key = (x, y + increment)  # shift position to down
                locked.append(new_key)
                locked.remove(key)

    return increment


# draws the upcoming piece
def draw_next_shape(piece, surface):
    font = pygame.font.Font(fontpath, 30)
    label = font.render('Next shape', 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '1':
                pygame.draw.rect(surface, piece.color,
                                 (start_x + j * block_size, start_y + i * block_size, block_size, block_size), 0)

    surface.blit(label, (start_x, start_y - 30))

    # pygame.display.update()


# draws the content of the window
def draw_window(surface, grid, score=0, last_score=0):
    surface.fill((0, 0, 0))  # fill the surface with black

    pygame.font.init()  # initialise font
    font = pygame.font.Font(fontpath_mario, 65, bold=True)
    label = font.render('TETRIS', 1, (255, 255, 255))  # initialise 'Tetris' text with white

    surface.blit(label, (
        (top_left_x + play_width / 2) - (label.get_width() / 2), 30))  # put surface on the center of the window

    # current score
    font = pygame.font.Font(fontpath, 30)
    label = font.render('SCORE   ' + str(score), 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    surface.blit(label, (start_x, start_y + 200))

    # last score
    label_hi = font.render('HIGHSCORE   ' + str(last_score), 1, (255, 255, 255))

    start_x_hi = top_left_x - 240
    start_y_hi = top_left_y + 200

    surface.blit(label_hi, (start_x_hi + 20, start_y_hi + 200))

    # draw content of the grid
    for i in range(row):
        for j in range(col):
            # pygame.draw.rect()
            # draw a rectangle shape
            # rect(Surface, color, Rect, width=0) -> Rect
            if grid[i][j] == 1:
                pygame.draw.rect(surface, (122, 122, 122),
                                 (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)

    # draw vertical and horizontal grid lines
    draw_grid(surface)

    # draw rectangular border around play area
    border_color = (255, 255, 255)
    pygame.draw.rect(surface, border_color, (top_left_x, top_left_y, play_width, play_height), 4)

    # pygame.display.update()


# update the score txt file with high score
def update_score(new_score):
    score = get_max_score()

    with open(filepath, 'w') as file:
        if new_score > score:
            file.write(str(new_score))
        else:
            file.write(str(score))


# get the high score from the file
def get_max_score():
    with open(filepath, 'r') as file:
        lines = file.readlines()  # reads all the lines and puts in a list
        score = int(lines[0].strip())  # remove \n

    return score


def main(window):
    locked_positions = {}
    create_grid(locked_positions)

    change_piece = False
    run2 = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.35
    level_time = 0
    score = 0
    last_score = get_max_score()

    while run2:
        # need to constantly make new grid as locked positions always change
        grid = create_grid(locked_positions)

        # helps run the same on every computer
        # add time since last tick() to fall_time
        fall_time += clock.get_rawtime()  # returns in milliseconds
        level_time += clock.get_rawtime()

        clock.tick()  # updates clock

        if level_time / 1000 > 5:  # make the difficulty harder every 10 seconds
            level_time = 0
            if fall_speed > 0.15:  # until fall speed is 0.15
                fall_speed -= 0.005

        if fall_time / 1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                # since only checking for down - either reached bottom or hit another piece
                # need to lock the piece position
                # need to generate new piece
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run2 = False
                pygame.display.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # rotate shape
                    current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)

                elif event.key == pygame.K_UP:
                    current_piece.y += 1
                    while valid_space(current_piece, grid):
                        current_piece.y += 1
                    else:
                        current_piece.y -= 1

        keys = pygame.key.get_pressed()

        if pygame.key.get_pressed()[pygame.K_DOWN]:
            time.sleep(0.1)
            current_piece.y += 1
            if not valid_space(current_piece, grid):
                current_piece.y -= 1

        elif pygame.key.get_pressed()[pygame.K_LEFT]:
            time.sleep(0.1)
            current_piece.x -= 1
            if not valid_space(current_piece, grid):
                current_piece.x += 1

        elif pygame.key.get_pressed()[pygame.K_RIGHT]:
            time.sleep(0.1)
            current_piece.x += 1
            if not valid_space(current_piece, grid):
                current_piece.x -= 1

        piece_pos = convert_shape_format(current_piece)

        # draw the piece on the grid by giving color in the piece locations
        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y >= 0:
                grid[y][x] = current_piece.color

        if change_piece:  # if the piece is locked
            for pos in piece_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color  # add the key and value in the dictionary
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            cleared_rows = clear_rows(grid, locked_positions)
            # increase score according to the Original BPS scoring system divided by 10 for big number purposes : https://tetris.wiki/Scoring#Original_BPS_scoring_system
            if cleared_rows == 1:
                score += 4
            elif cleared_rows == 2:
                score += 10
            elif cleared_rows == 3:
                score += 30
            elif cleared_rows == 4:
                score += 120
            update_score(score)

            if last_score < score:
                last_score = score

        draw_window(window, grid, score, last_score)
        draw_next_shape(next_piece, window)
        pygame.display.update()

        if check_lost(locked_positions):
            run2 = False

    draw_text_middle('You Lost', 40, (255, 255, 255), window)
    pygame.display.update()
    pygame.time.delay(2000)  # wait for 2 seconds
    pygame.quit()
    run = False


def main_menu(window):
    run = True
    while run:
        draw_text_middle('Press any key to begin', 50, (255, 255, 255), window)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                main(window)
                run = False

    pygame.quit()


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, win=None):
        super(TetrisEnv).__init__()
        self.reward_range = (0, 999_999)
        obs_spaces = {
            'grid': spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.uint8),
            'piece': spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.uint8),
            'pos': spaces.Box(low=0, high=np.array([10, 20]), shape=(2,), dtype=np.uint8)
        }
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(obs_spaces)
        self.locked_positions = []
        self.score = 0
        self.window = win
        self.grid = grid_for_ai_with_list()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        reward, done = self._take_action(action)

        self.current_step += 1

        return self._next_observation(), reward, done, {}

    def _take_action(self, action):
        reward = 0
        done = False
        grid = create_grid(self.locked_positions)

        # helps run the same on every computer
        # add time since last tick() to fall_time
        self.fall_time += self.clock.get_rawtime()  # returns in milliseconds
        self.level_time += self.clock.get_rawtime()

        self.clock.tick()  # updates clock

        if self.level_time / 1000 > 5:  # make the difficulty harder every 10 seconds
            self.level_time = 0
            if self.fall_speed > 0.15:  # until fall speed is 0.15
                self.fall_speed -= 0.005

        if self.fall_time / 1000 > self.fall_speed:
            self.fall_time = 0
            self.current_piece.y += 1
            if not valid_space(self.current_piece, grid) and self.current_piece.y > 0:
                self.current_piece.y -= 1
                # since only checking for down - either reached bottom or hit another piece
                # need to lock the piece position
                # need to generate new piece
                self.change_piece = True

        if action == 0:
            # Left
            self.current_piece.x -= 1
            if not valid_space(self.current_piece, grid):
                self.current_piece.x += 1

        elif action == 1:
            # Right
            self.current_piece.x += 1
            if not valid_space(self.current_piece, grid):
                self.current_piece.x -= 1

        elif action == 2:
            # Down
            self.current_piece.y += 1
            if not valid_space(self.current_piece, grid):
                self.current_piece.y -= 1

        elif action == 3:
            # Rotate
            self.current_piece.rotation = self.current_piece.rotation + 1 % len(self.current_piece.shape)
            if not valid_space(self.current_piece, grid):
                self.current_piece.rotation = self.current_piece.rotation - 1 % len(self.current_piece.shape)

        piece_pos = convert_shape_format(self.current_piece)

        if self.change_piece:  # if the piece is locked
            for pos in piece_pos:
                p = (pos[0], pos[1])
                self.locked_positions.append(p)  # add the key and value in the dictionary
            self.grid = grid_for_ai_with_list(self.locked_positions)
            self.current_piece = self.next_piece
            self.next_piece = get_shape()
            self.change_piece = False
            cleared_rows = clear_rows(self.grid, self.locked_positions)
            # increase score according to the Original BPS scoring system divided by 10 for big number purposes : https://tetris.wiki/Scoring#Original_BPS_scoring_system
            dic = {0: 0, 1: 4, 2: 10, 3: 30, 4: 120}
            reward = dic[cleared_rows]
            self.score += reward
            update_score(self.score)

        if check_lost(self.locked_positions):
            done = True
            reward -= 10
        return reward, done

    def reset(self):
        self.locked_positions = []
        grid_for_ai_with_list(self.locked_positions)

        self.change_piece = False
        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 0.35
        self.level_time = 0
        self.score = 0

        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human'):
        draw_window(self.window, self.grid, self.score)
        for i, j in convert_shape_format(self.current_piece):
            pygame.draw.rect(self.window, self.current_piece.color,
                             (top_left_x + i * block_size, top_left_y + j * block_size, block_size, block_size), 0)
        draw_next_shape(self.next_piece, self.window)
        pygame.display.update()

    def _next_observation(self):
        shape = self.current_piece.shape[0].copy()
        for i, row in enumerate(shape):
            shape[i] = list(map(int, list(row)))
        obs = [np.array(grid_for_ai(self.locked_positions), dtype=np.int8), np.array(shape, dtype=np.int8), np.array([self.current_piece.x, self.current_piece.y], dtype=np.int8)]
        return obs


def build_model(states, actions):
    model_grid = Sequential()
    model_grid.add(Flatten(input_shape=(1,) + states['grid'].shape, name='grid', dtype=np.int8))
    model_grid_input = Input(shape=(1,) + states['grid'].shape, name='grid', dtype=np.int8)
    model_grid_encoded = model_grid(model_grid_input)

    model_piece = Sequential()
    model_piece.add(Flatten(input_shape=(1,) + states['piece'].shape, name='piece', dtype=np.int8))
    model_piece_input = Input(shape=(1,) + states['piece'].shape, name='piece', dtype=np.int8)
    model_piece_encoded = model_piece(model_piece_input)

    model_pos = Sequential()
    model_pos.add(Flatten(input_shape=(1,) + states['pos'].shape, name='pos', dtype=np.int8))
    model_pos_input = Input(shape=(1,) + states['pos'].shape, name='pos', dtype=np.int8)
    model_pos_encoded = model_pos(model_pos_input)

    con = concatenate([model_grid_encoded, model_piece_encoded, model_pos_encoded])

    hidden = Dense(24, activation='relu')(con)
    for _ in range(2):
        hidden = Dense(24, activation='relu')(hidden)
    output = Dense(actions, activation='linear')(hidden)
    model_final = Model(inputs=[model_grid_input, model_piece_input, model_pos_input], outputs=output)
    return model_final


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


if __name__ == '__main__':
    vizualize = False
    if vizualize:
        win = pygame.display.set_mode((s_width, s_height))
        pygame.display.set_caption('Tetris')
        env = TetrisEnv(win)
    else:
        env = TetrisEnv()
    states = env.observation_space.spaces
    print(states)
    actions = env.action_space.n
    model = build_model(states, actions)
    plot_model(model, to_file='model.png', show_shapes=True)
    print(model.summary())
    dqn = build_agent(model, actions)
    dqn.processor = MultiInputProcessor(3)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.load_weights("tests_weights.h5f")
    dqn.fit(env, nb_episodes=1000, visualize=vizualize, verbose=1)
    dqn.save_weights("tests_weights.h5f", overwrite=True)
    """dqn.load_weights("tests_weights.h5f")
    print(dqn)
    if not vizualize:
        win = pygame.display.set_mode((s_width, s_height))
        pygame.display.set_caption('Tetris')
        env = TetrisEnv(win)
    dqn.test(env, nb_episodes=5, visualize=True)"""