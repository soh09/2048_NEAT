import pickle
import random

try:
    with open("C:\\Users\\hirot\\Documents\\2048_NEAT\\python_2048\\2048_lut.pkl", "rb") as f:
        data = pickle.load(f)
        MOVES_LEFT = data["left"]
        MOVES_RIGHT = data["right"]
        LEFT_SCORES = data["left_score"]
        RIGHT_SCORES = data["right_score"]
        ROW_INFO_TABLE = data["row_info_table"]
        print("luts loaded successfully.")
except FileNotFoundError:
    print("Error: '2048_lut.pkl' not found. Please run generate_lut.py first.")
    # Optional: You could call the generation function here as a fallback
    raise(Exception)

CHOICES = list(MOVES_LEFT.keys())

def print_2048(rows):
    # 1. Calculate the max width needed for the numbers
    max_w = 0
    for row in rows:
        for e in row:
            if len(str(e)) > max_w:
                max_w = len(str(e))
    
    # 2. Calculate the total width of the box to make borders match
    # (cols * (number_width + 1 for padding)) + (cols - 1 for spaces between)
    cols = len(rows[0])
    line_width = (cols * (max_w + 1)) + (cols - 1)

    # 3. Print the Box
    print(' ' + '_' * line_width + ' ')  # Dynamic Top Border
    for row in rows:
        print(f'|{" ".join(f"{str(e):>{max_w + 1}}" for e in row)}|')
    print(' ' + '‾' * line_width + ' ')  # Dynamic Bottom Border


num_to_4bit = {
    2**i if i >= 1 else 0: format(i, 'b').zfill(4) for i in range(0, 16)
}

fourbit_to_num = {
    format(i, 'b').zfill(4): 2**i if i >= 1 else 0 for i in range(0, 16)
}

def encode_list_to_str(l):
    s = "".join(num_to_4bit[n] for n in l)
    return s

def decode_str_to_list(s):
    s = s.zfill(16)
    a, b, c, d = [s[i*4:(i+1)*4] for i in range(4)]
    return [fourbit_to_num[a], fourbit_to_num[b], fourbit_to_num[c], fourbit_to_num[d]]

def transpose(board):
    # first, swap the top right 2 * 2 and bottom left 2 * 2
    # num = a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 d1 d2 d3 d4
    # swap a3 a4 b3 b4 and c1 c2 d1 d2

    top_right_mask = ((0b11111111 << 16) | 0b11111111) << 32
    bottom_left_mask = (0b11111111 << 24) | (0b11111111 << 8)

    top_right = (board & top_right_mask) >> 32
    bottom_left = (board & bottom_left_mask) >> 8

    diff = top_right ^ bottom_left

    mask = diff << 32 | diff << 8

    board ^= mask
    
    # now, tranpose the individual 2*2
    even_mask = (0b0000111100001111 << 48) | (0b0000111100001111 << 16)
    odd_mask = (0b1111000011110000 << 32) | 0b1111000011110000

    evens = (board & even_mask) >> 16
    odds = (board & odd_mask) >> 4

    diff = evens ^ odds

    mask = diff << 4 | diff << 16

    board ^= mask

    return board


def make_test_board():
    n = 0
    for _ in range(4):
        n <<= 16
        row = random.choice(CHOICES)
        # print(row)
        n |= row
    return n

def int_to_board(num):
    rows = get_rows(num)
    rows = [decode_str_to_list(format(row, 'b')) for row in rows] # 2d array format
    print_2048(rows)

def get_rows(num):
    mask = 0b1111111111111111
    d = num & mask
    c = (num >> 16) & mask
    b = (num >> 32) & mask
    a = (num >> 48) & mask
    return a, b, c, d

# move left -> break into 16 bit chunks look up table
def move_left(num):
    # need to handle scoring too
    a, b, c, d = get_rows(num)
    res = 0
    res |= MOVES_LEFT[a]
    res <<= 16
    res |= MOVES_LEFT[b]
    res <<= 16
    res |= MOVES_LEFT[c]
    res <<= 16
    res |= MOVES_LEFT[d]

    score = LEFT_SCORES[a] + LEFT_SCORES[b] + LEFT_SCORES[c] + LEFT_SCORES[d]
    return res, score


# move right -> reverse, then look up table
# given a int thats 64 bits -> break into 16 bit chunks, reverse. look up each, then combine it back
def move_right(num):
    a, b, c, d = get_rows(num)

    res = 0
    res |= MOVES_RIGHT[a]
    res <<= 16
    res |= MOVES_RIGHT[b]
    res <<= 16
    res |= MOVES_RIGHT[c]
    res <<= 16
    res |= MOVES_RIGHT[d]

    score = RIGHT_SCORES[a] + RIGHT_SCORES[b] + RIGHT_SCORES[c] + RIGHT_SCORES[d]
    return res, score


def move_up(num):
    num = transpose(num)
    num, score = move_left(num)
    num = transpose(num)
    return num, score

def move_down(num):
    num = transpose(num)
    num, score = move_right(num)
    num = transpose(num)
    return num, score


def new_game():
    pos = random.choices([i for i in range(16)], k=2)
    while pos[0] == pos[1]:
        pos = random.choices([i for i in range(16)], k = 2)
    return 0b01 << pos[0] * 4 | 0b01 << pos[1] * 4

def get_game_state(board):
    # Unpack rows
    r0 = board & 0xFFFF
    r1 = (board >> 16) & 0xFFFF
    r2 = (board >> 32) & 0xFFFF
    r3 = (board >> 48) & 0xFFFF
    
    rows = [r0, r1, r2, r3]

    # 1. Check for WIN (Priority 1)
    # We can check all rows for 2048 immediately
    for r in rows:
        if ROW_INFO_TABLE[r]['win']:
            return 'win'

    # 2. Check for "Not Over" in ROWS
    # If any row has a valid move (empty or merge), the game continues.
    for r in rows:
        if ROW_INFO_TABLE[r]['move']:
            return 'not over'
            
    # 3. Check for "Not Over" in COLS
    # If we are here, it means all ROWS are full and have no merges.
    # But we might have a vertical merge available.
    
    # Transpose to turn columns into rows
    t_board = transpose(board)
    
    c0 = t_board & 0xFFFF
    c1 = (t_board >> 16) & 0xFFFF
    c2 = (t_board >> 32) & 0xFFFF
    c3 = (t_board >> 48) & 0xFFFF
    
    cols = [c0, c1, c2, c3]
    
    for c in cols:
        # Note: We don't need to check 'win' here, we already checked every tile in step 1.
        # We only care if a move is possible.
        if ROW_INFO_TABLE[c]['move']:
            return 'not over'

    # 4. If no horizontal or vertical moves...
    return 'lose'

def generate_next(board):
    empty_list = []
    for i in range(16):
        if (board >> (i * 4)) & 0xF == 0:
            empty_list.append(i)
    if not empty_list:
        return board
    
    pos = random.choice(empty_list)
    return board | 0b0001 << pos * 4

MOVE_DICT = {
    'down': move_down,
    'up': move_up,
    'left': move_left,
    'right': move_right
}

class Game:
    def __init__(self):
        self.board = new_game()
        self.filled_in = 2
        self.combined = 1
        # self.max = 0

    def get_max_item(self): # this is the fitness of the neural net
        pass

        # keep track of max score (return of move), this makes tracking this easy

    def get_numbers_combined(self):
        return self.combined
    
    def get_reward(self, reward_type):
        if reward_type == 'MAX VALUE':
            return self.get_max_item()
        
        if reward_type == 'COMBINED NUMBERS':
            return self.combined
    
    def get_board(self):
        return [
            (self.board >> 60) & 0xF, (self.board >> 56) & 0xF, (self.board >> 52) & 0xF, (self.board >> 48) & 0xF, # Row 0
            (self.board >> 44) & 0xF, (self.board >> 40) & 0xF, (self.board >> 36) & 0xF, (self.board >> 32) & 0xF, # Row 1
            (self.board >> 28) & 0xF, (self.board >> 24) & 0xF, (self.board >> 20) & 0xF, (self.board >> 16) & 0xF, # Row 2
            (self.board >> 12) & 0xF, (self.board >> 8) & 0xF,  (self.board >> 4) & 0xF,  self.board & 0xF        # Row 3
        ]
    
    # # unused method, iteration #1 of this function
    # def do_next_move(self, move: str, debug = False): # move will be provided by neural net
    #     self.mat, _ = MOVE_DICT[move](self.mat, debug)
    #     state = game_state(self.mat)
    #     return state
    
    def do_next_move_and_track(self, move: str, debug = False):
        if debug:
            int_to_board(self.board)
        new_board, score = MOVE_DICT[move](self.board)
        if debug:
            int_to_board(new_board)

        self.combined += score
        # self.filled_in = new_filled_in
        self.board = new_board

        state = get_game_state(self.board)
        if debug:
            print(state)
        return state
    
    def generate_next(self):
        self.board = generate_next(self.board)
        # self.filled_in += 1 # generate_next() always adds one number to the board


    def __repr__(self):
        rows_ints = get_rows(self.board)
        
        matrix = []
        for r in rows_ints:
            # Extract nibbles (4-bit chunks)
            # (r >> 12) is the left-most tile in the row
            row_nibbles = [
                (r >> 12) & 0xF,
                (r >> 8) & 0xF,
                (r >> 4) & 0xF,
                r & 0xF
            ]

            matrix.append([2**n if n > 0 else 0 for n in row_nibbles])

        flat_vals = [str(n) for row in matrix for n in row]
        max_w = max(len(s) for s in flat_vals)
        
        cols = 4
        line_width = (cols * (max_w + 1)) + (cols - 1)

        # Build the string output
        output = []
        output.append(' ' + '_' * line_width + ' ')  # Top border
        
        for row in matrix:
            inner = " ".join(f"{str(e):>{max_w + 1}}" for e in row)
            output.append(f'|{inner}|')
            
        output.append(' ' + '‾' * line_width + ' ')  # Bottom border
        
        return "\n".join(output)

        


