# created by soh09 with most of the code borrowed from original 2048-python repository

import random
GRID_LEN = 4


def gen():
    return random.randint(0, GRID_LEN - 1)

def new_game(n):
    matrix = []
    for i in range(n):
        matrix.append([0] * n)
    matrix = add_two(matrix)
    matrix = add_two(matrix)
    return matrix

def add_two(mat):
    a = random.randint(0, len(mat)-1)
    b = random.randint(0, len(mat)-1)
    while mat[a][b] != 0:
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    mat[a][b] = 2
    return mat

def game_state(mat):
    # check for win cell
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 2048:
                return 'win'
    # check for any zero entries
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
    # check for same cells that touch each other
    for i in range(len(mat)-1):
        # intentionally reduced to check the row on the right and below
        # more elegant to use exceptions but most likely this will be their solution
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new

def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new

def cover_up(mat):
    new = []
    for j in range(GRID_LEN):
        partial_new = []
        for i in range(GRID_LEN):
            partial_new.append(0)
        new.append(partial_new)
    done = False
    for i in range(GRID_LEN):
        count = 0
        for j in range(GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done

def merge(mat, done):
    for i in range(GRID_LEN):
        for j in range(GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                done = True
    return mat, done

def up(game, debug):
    if debug:
        print("up")
    # return matrix after shifting up
    game = transpose(game)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(game)
    return game, done

def down(game, debug):
    if debug:
        print("down")
    # return matrix after shifting down
    game = reverse(transpose(game))
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return game, done

def left(game, debug):
    if debug:
        print("left")
    # return matrix after shifting left
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    return game, done

def right(game, debug):
    if debug:
        print("right")
    # return matrix after shifting right
    game = reverse(game)
    game, done = cover_up(game)
    game, done = merge(game, done)
    game = cover_up(game)[0]
    game = reverse(game)
    return game, done

def generate_next(game):
    # this function only ever adds one number at a time
    index = (gen(), gen())
    while game[index[0]][index[1]] != 0:
        index = (gen(), gen())
    game[index[0]][index[1]] = 2



class Game:
    move_d = {
        'down': down,
        'up': up,
        'left': left,
        'right': right
    }

    def __init__(self):
        self.mat = new_game(GRID_LEN)
        self.combined = 1 # start with 1 to avoid zero division later in fitness adjustment or allocating # of offsprings
        self.filled_in = 0 # number of non-zero numbers present in previous board

        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                if self.mat[i][j] != 0:
                    self.filled_in += 1

    def get_max_item(self): # this is the fitness of the neural net
        return max(max(row) for row in self.mat)

    def get_numbers_combined(self):
        return self.combined
    
    def get_reward(self, reward_type):
        if reward_type == 'MAX VALUE':
            return self.get_max_item()
        
        if reward_type == 'COMBINED NUMBERS':
            return self.get_numbers_combined()
    
    def get_board(self): # this will be the input to the neural net
        l = []
        for row in self.mat:
            l.extend(row)
        return l
    
    def do_next_move(self, move: str, debug = False): # move will be provided by neural net
        self.mat, _ = Game.move_d[move](self.mat, debug)
        state = game_state(self.mat)
        return state
    
    def do_next_move_and_track(self, move: str, debug = False):
        new_mat, _ = Game.move_d[move](self.mat, debug)

        # calculate the number of squares that were combined
        new_filled_in = 0
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                if new_mat[i][j] != 0:
                    new_filled_in += 1
        diff = self.filled_in - new_filled_in
        self.combined += diff
        self.filled_in = new_filled_in
        self.mat = new_mat

        state = game_state(self.mat)
        return state
    
    def generate_next(self):
        generate_next(self.mat)
        self.filled_in += 1 # generate_next() always adds one number to the board


    def __repr__(self):
        return '[' + '\n '.join(['[' + ', '.join([str(i) for i in row]) + ']' for row in self.mat]) + ']'