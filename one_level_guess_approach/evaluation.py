import torch
import numpy as np
from matplotlib.pyplot import *

def cumulative_histogram(x, y, z, perc):

    x, y, z = np.array(sorted(x)), np.array(sorted(y)), np.array(sorted(z))
    n_x, n_y, n_z = len(x), len(y), len(z)

    cumulative_x = np.arange(1, n_x + 1) / n_x * 100
    cumulative_y = np.arange(1, n_y + 1) / n_y * 100
    cumulative_z = np.arange(1, n_z + 1) / n_z * 100

    plot(x, cumulative_x)
    plot(y, cumulative_y)
    plot(z, cumulative_z)

    xscale('log'),
    xlabel("Guesses until Success"),
    ylabel("Cumulative Percentage (%)"), title(f"Cumulative Histogram ({n_x})"),
    grid(True), xlim(0.9, 200),  # Adjust x-axis limits
    x_ticks = np.concatenate((np.arange(1, 11), [15, 20, 50, 100, 150]))
    xticks(x_ticks, x_ticks)

    # Find and plot 95% mark for x (red)
    index_95_x = np.where(cumulative_x >= perc)[0]
    if len(index_95_x) > 0:
        x_95 = x[index_95_x[0]]
        y_95 = cumulative_x[index_95_x[0]]

        hlines(y=perc, xmin=0, xmax=x_95, color='blue', linestyle='-.', linewidth=0.7)
        vlines(x=x_95, ymin=0, ymax=y_95, color='blue', linestyle='--', linewidth=0.7)
        plot(x_95, y_95, 'bo', markersize=4)  # Red circle

    # Find and plot 95% mark for y (green)
    index_95_y = np.where(cumulative_y >= perc)[0]
    if len(index_95_y) > 0:
        y_95_x = y[index_95_y[0]]
        y_95_y = cumulative_y[index_95_y[0]]

        hlines(y=perc, xmin=0, xmax=y_95_x, color='orange', linestyle='--', linewidth=0.7)
        vlines(x=y_95_x, ymin=0, ymax=y_95_y, color='orange', linestyle='--', linewidth=0.7)
        plot(y_95_x, y_95_y, 'o', color='orange', markersize=4)  # Green circle


    # Find and plot 95% mark for y (green)
    index_95_z = np.where(cumulative_z >= perc)[0]
    if len(index_95_z) > 0:
        z_95_x = z[index_95_z[0]]
        z_95_y = cumulative_z[index_95_z[0]]

        hlines(y=perc, xmin=0, xmax=y_95_x, color='green', linestyle=':', linewidth=0.7)
        vlines(x=z_95_x, ymin=0, ymax=z_95_y, color='green', linestyle='--', linewidth=0.7)
        plot(z_95_x, z_95_y, 'o', color='green', markersize=4)  # Green circle

    legend([f"upper bound ({str(perc)}%): {x_95:2.2f} guesses", f"lower bound ({str(perc)}%): {y_95_x:2.2f} guesses", f"ours ({str(perc)}%): {z_95_x:2.2f} guesses"], loc='lower right')
    show()


def get_cell_acc(X,Y,z):
    '''
    Computes cell accuracy for the scratchpad part.
    Every token should output the correct value of the cell in the corresponding position.
    '''
    with torch.no_grad():
        tot_cell_acc = []
        pos_103 = torch.where(X == 103)[1].reshape(-1, 81)
        for j in range(X.shape[0]):
            ncorrect = 0
            for k in pos_103[j]:
                log_softmax_103_k = z[j][k]
                targets_103_k = Y[j][k]

                v_pred = torch.argmax(log_softmax_103_k).item()
                v_true = torch.argmax(targets_103_k.float()).item()
                #print(v_pred, v_true)
                if v_pred == v_true: ncorrect += 1

            cell_acc = ncorrect / 81
            #print(cell_acc)
            tot_cell_acc.append(cell_acc)

    return np.mean(tot_cell_acc)

def parse_moves(moves):
    """Parses the list of 3-digit moves into a Sudoku board representation."""
    board = [[0] * 9 for _ in range(9)]
    for move in moves:
        r, c, v = (move // 100) - 1, ((move % 100) // 10) - 1, (move % 10)
        board[r][c] = v
    return board

def get_possible_moves(board):
    """Finds all valid moves that do not violate Sudoku rules."""
    possible_moves = []
    npossible_moves = 0
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:  # Empty cell
                used_values = set()

                # Check row and column
                for i in range(9):
                    used_values.add(board[r][i])  # Row
                    used_values.add(board[i][c])  # Column

                # Check 3x3 box
                box_r, box_c = (r // 3) * 3, (c // 3) * 3
                for i in range(3):
                    for j in range(3):
                        used_values.add(board[box_r + i][box_c + j])

                # Possible values
                for v in range(1, 10):
                    if v not in used_values:
                        #possible_moves.append((r + 1) * 100 + (c + 1) * 10 + v)
                        npossible_moves += 1

    return npossible_moves


def update_lower_bound(X, Y, lower_bound_list):
    eps = 1e-6
    v = 101
    Nscratchpad = 0
    with torch.no_grad():
        pos_101 = torch.where(X == v)[1].reshape(-1, Nscratchpad+1)
        for j in range(X.shape[0]):
            posi_first, posi_last = pos_101[j][0], pos_101[j][-1]
            n_magic_cells = Y[j][posi_last].sum()
            board_moves = X[j][:posi_first].tolist()
            n_allmoves = get_possible_moves(parse_moves(board_moves))
            lower_bound_list.append( (1 / (n_magic_cells / (n_allmoves + eps) + eps)).item() )

def update_upper_bound(X, Y, upper_bound_list):
    eps = 1e-6
    v = 101
    Nscratchpad = 0
    with torch.no_grad():
        pos_101 = torch.where(X == v)[1].reshape(-1, Nscratchpad+1)
        for j in range(X.shape[0]):
            posi_first, posi_last = pos_101[j][0], pos_101[j][-1]
            n_magic_cells = Y[j][posi_last].sum()
            n_empty_cells = len(X[j]) - posi_first - 1 - Nscratchpad - 81
            upper_bound_list.append( (1 / (n_magic_cells / (n_empty_cells + eps) + eps)).item() )

def get_idx(moves):
    idxs = []
    for movei in moves:
        r,c,v = movei//100, (movei%100)//10, movei%10
        idx = (r-1)*9 + c
        idxs.append(idx)
    return idxs


def update_inv_prob(X, Y, probs, inv_probs_list):
    eps = 1e-6
    v = 101
    Nscratchpad = 0
    with torch.no_grad():
        pos_101 = torch.where(X == v)[1].reshape(-1, Nscratchpad+1)
        for j in range(X.shape[0]):
            probs_101 = probs[j][pos_101[j][-1]]
            targets_101 = Y[j][pos_101[j][-1]]
            inv_probs_list.append( (1 / torch.clamp((probs_101[targets_101].sum() + eps), min=0.001)).item() )
