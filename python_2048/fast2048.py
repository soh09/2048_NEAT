import pickle

def shift_left(l):
    new = [0, 0, 0, 0]
    new_i = 0
    for num in l:
        if num != 0:
            new[new_i] = num
            new_i += 1
    return new

def merge(l):
    for i in range(len(l) - 1):
        if l[i] == l[i+1] and l[i] != 32768:
            l[i] = l[i] * 2
            l[i+1] = 0
    return l


def do_left(l):
    return shift_left(merge(shift_left(l)))


try:
    with open("C:\Users\hirot\Documents\2048_NEAT\python_2048\2048_lut.pkl", "rb") as f:
        data = pickle.load(f)
        MOVES_LEFT = data["row"]
        SCORES = data["score"]
        print("2048 LUT loaded successfully.")
except FileNotFoundError:
    print("Error: '2048_lut.pkl' not found. Create look up table or use correct path.")
    # Optional: You could call the generation function here as a fallback
    exit()

class Game:
    def __init__(self):
        


