import os
from gdown import download
import numpy as np
from itertools import permutations

filename = "sudoku_processed.csv.gz"
if not os.path.exists(filename):
    download(id="1eGvm9b6bd2LaqSnQ7eUol5V8tt59M2_6", output=filename, quiet=False)


elements = np.arange(9)
all_permutations = np.array(list(permutations(elements)))
Q = np.loadtxt(filename, delimiter=',',dtype=int)

def inverse(p):
    p = p.reshape(-1)
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    return inv

def toNumber(p):
    return int(''.join(map(str,list(p.reshape(-1)))))

def fromNumber(n):
    res = []
    for i in range(9):
        res.append(n%10)
        n //= 10
    return np.array( list(reversed(res)) ).reshape(-1)

all_permutations_no = np.array( [toNumber(p) for p in all_permutations] )

def board_generate(c):
    final_perm_idx = (c-1) // 18383222420692992
    c = (c-1) % 18383222420692992 + 1
    if final_perm_idx < 0 or final_perm_idx >= len(all_permutations): return False

    perm = all_permutations[ final_perm_idx ]
    row_perm = np.arange(9)
    col_perm = np.arange(9)

    Final = np.zeros((9,9), dtype=int)
    Final[ :3,  :3 ] = perm[ np.arange(9).reshape(3,3) ]

    for i in range(1,9):
        idx = np.searchsorted(Q[:,0], c) - 1

        Curr = np.zeros((9,9), dtype=int)
        Cell = fromNumber(Q[idx,2]).reshape(3,3)
        Curr[ (i // 3 * 3):(i // 3 * 3 + 3),  (i % 3 * 3):(i % 3 * 3 + 3) ] = perm[ Cell ]
        Final += Curr[ row_perm[:,None], col_perm[None,:] ]

        c += Q[ idx, 1 ] - Q[ idx, 0 ]
        curr_perm, curr_row_perm, curr_col_perm = \
            fromNumber( Q[idx,3] ), fromNumber( Q[idx,4] ), fromNumber( Q[idx,5] )
        row_perm = curr_row_perm[ row_perm ]
        col_perm = curr_col_perm[ col_perm ]
        perm = perm[ curr_perm ]
    return ''.join( map(str,Final.reshape(-1) + 1) )


def getIndex(n):
    return np.searchsorted(Q[:,0], n)


def encode(sudoku):
    if type(sudoku) == bool: return False;
    if type(sudoku) == str:
        sudoku = np.array( list( map(int, sudoku) ) )
    Final = sudoku.reshape(9,9) - 1
    Cell = np.zeros((3,3), dtype=int)
    Cell = Final[:3,:3]

    pid = np.searchsorted(all_permutations_no, toNumber(Cell))
    perm = inverse( Cell.reshape(-1) )
    Curr = perm[ Final ]
    c = 0
    cnt_start = 0
    idx_start = getIndex(cnt_start)

    for i in range(1,9):
        Cell = Curr[ (i // 3 * 3):(i // 3 * 3 + 3),  (i % 3 * 3):(i % 3 * 3 + 3) ]
        idx = idx_start + np.where( Q[idx_start:,2] == toNumber(Cell))[0][0]
        c += Q[idx, 0] - cnt_start

        cnt_start = Q[idx, 1]
        idx_start = getIndex(cnt_start)

        perm, row_perm, col_perm = \
                inverse( fromNumber( Q[idx,3] ) ), inverse( fromNumber( Q[idx,4] ) ), inverse( fromNumber( Q[idx,5] ) )
        #print(perm, row_perm, col_perm)
        Curr = perm[ Curr[ row_perm[:,None], col_perm[None,:] ] ]

    return int(pid) * 18383222420692992 + int(c) + 1