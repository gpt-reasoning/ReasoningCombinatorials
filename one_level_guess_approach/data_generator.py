import numpy as np
import torch
from torch import tensor as tt
from ReasoningCombinatorials.one_level_guess_approach.board_enumeration import *
from ReasoningCombinatorials.one_level_guess_approach.sudoku_generator import *
from ReasoningCombinatorials.one_level_guess_approach.single_guess import *

def moves_to_string(moves):
    '''
    converts a sequence of 3-digit moves to a sudoku string
    '''
    board = ['.'] * 81
    for move in moves:
        r, c, v = (move // 100) - 1, ((move % 100) // 10) - 1, (move % 10)
        board[r * 9 + c] = str(v)
    return "".join(board)

def string_to_moves(sudoku_string):
    '''
    converts a sudoku string to a sequence of 3-digit moves
    '''
    moves, grid = [], []
    for i in range(9):
        grid.append(list(sudoku_string[i * 9: (i + 1) * 9]))

    for r in range(9):
        for c in range(9):
            if grid[r][c] != '.':
                v = int(grid[r][c])
                moves.append((r + 1) * 100 + (c + 1) * 10 + v)

    return moves

class SudokuDataset(torch.utils.data.Dataset):

    def __len__(self):
        return 9999  # Some large number to allow sampling

    def __getitem__(self,idx):

        # generate Sudoku boards and their solution paths until a <guess> is included in the path
        has_guess = False
        while not has_guess:

            # generate an unsolved Sudoku board
            sudoku_id = np.random.randint(18383222420692992) +  np.random.randint(1*2*3*4*5*6*7*8*9) * 18383222420692992
            solution = board_generate( sudoku_id )
            sudoku_string = puzzle_generate(solution, np.random.permutation(81))

            # get the solution path
            l,X,Y = solve_path(sudoku_string)
            if 101 in X: has_guess=True

        posi = torch.where(X == 101)[0] # length of sequence until first guess
        X[posi+1:] = 0                  # remove all moves of sequence after 101 (rules end)

        # For the Guess-node (after rules-end (101)) add 81 thinking tokens (scratchpad) & their corresponding target vectors (y_scratchpad)
        # each thinking token corresponds to a particular cell in sudoku board. Its target is the corresponding correct value for that cell
        # all thinking tokens have value 103
        sol_moves = string_to_moves(solution)
        y_scratchpad = torch.zeros(81,1000)
        for movei in sol_moves:
            r,c,v = movei//100, (movei%100)//10, movei%10
            idx = (r-1)*9 + c
            y_scratchpad[idx-1][v] = 1

        scratchpad = tt([103]).repeat(81)
        X = torch.concat((X[:posi+1], scratchpad, X[posi+1:]), dim=-1)
        Y = torch.concat((Y[:posi+1], y_scratchpad, Y[posi+1:]), dim=0)

        return l,X,Y