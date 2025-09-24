
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define SIZE 9
#define BOX_SIZE 3
#define GROUPS 4
#define NEIGHBORS 20
#define NUM_CELLS (SIZE * SIZE)

int is_invalid;
int neighbors[NUM_CELLS][NEIGHBORS];
int board[NUM_CELLS];
int candidates[GROUPS][SIZE][SIZE];

int box_index(int box, int i) {
    int box_row = (box / BOX_SIZE) * BOX_SIZE; // Starting row of the box
    int box_col = (box % BOX_SIZE) * BOX_SIZE; // Starting column of the box
    int row = box_row + (i / BOX_SIZE); // Row within the box
    int col = box_col + (i % BOX_SIZE); // Column within the box
    return row * SIZE + col; // Convert (row, col) to 1D index
}
int gindex(int group, int a, int b, int c) {
    if(group == 0) return a*SIZE+b;
    if(group == 1) return b*SIZE+c;
    if(group == 2) return c*SIZE+b;
    return box_index(b,c);
}
int gvalue(int group, int a, int b, int c) {
    if(group == 0) return c;
    return a;
}
int to_position(int idx) {return (idx/9+1)*10+idx%9+1;}

void initialize() {
    is_invalid = 0;
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        board[idx] = 0;
    }
    for (int g=0;g<GROUPS;g++) {
        for (int a=0;a<SIZE;a++) {
            for (int b=0;b<SIZE;b++) {
                candidates[g][a][b] = (1<<SIZE)-1;
            }
        }
    }
}

void init_neighbors() {
    for(int idx=0;idx<NUM_CELLS;idx++) {
        int row = idx / SIZE;
        int col = idx % SIZE;
        int box = (row / BOX_SIZE) * BOX_SIZE + (col / BOX_SIZE);
        int boxi = (row % BOX_SIZE) * BOX_SIZE + (col % BOX_SIZE);
        int c = 0;
        for(int ncol=0;ncol<SIZE;ncol++) {
            if(ncol == col) continue;
            int nidx = row*SIZE + ncol;
            neighbors[idx][c++] = nidx;
        }
        for(int nrow=0;nrow<SIZE;nrow++) {
            if(nrow == row) continue;
            int nidx = nrow*SIZE + col;
            neighbors[idx][c++] = nidx;
        }
        for(int nboxi=0;nboxi<SIZE;nboxi++) {
            int nidx = box_index(box, nboxi);
            if( nidx / SIZE == row) continue;
            if( nidx % SIZE == col) continue;
            neighbors[idx][c++] = nidx;
        }
    }
}

int checkvalid() {
    int r[9], c[9], b[9];
    if(is_invalid) return 0;
    for(int i=0;i<9;i++) r[i] = c[i] = b[i] = 0;

    for (int idx = 0; idx < NUM_CELLS; idx++) {
        int row = idx / SIZE;
        int col = idx % SIZE;
        int box = (row / BOX_SIZE) * BOX_SIZE + (col / BOX_SIZE);
        int val = board[idx];
        if( val < 1 || val > 9) return 0;
        if( r[row] & (1<<val) ) return 0;
        if( c[col] & (1<<val) ) return 0;
        if( b[box] & (1<<val) ) return 0;

        r[row] |= (1<<val);
        c[col] |= (1<<val);
        b[box] |= (1<<val);
    }
    return 1;
}

int fill_value(int idx, int val) {
    int row = idx / SIZE;
    int col = idx % SIZE;
    int box = (row / BOX_SIZE) * BOX_SIZE + (col / BOX_SIZE);
    int boxi = (row % BOX_SIZE) * BOX_SIZE + (col % BOX_SIZE);
    int err = 0;

    board[idx] = val+1;

    if( ! (candidates[0][ row ][ col ] & (1<<val)) ) err = 1;
    if( ! (candidates[1][ val ][ row ] & (1<<col)) ) err = 1;
    if( ! (candidates[2][ val ][ col ] & (1<<row)) ) err = 1;
    if( ! (candidates[3][ val ][ box ] & (1<<boxi)) ) err = 1;

    candidates[0][row][col] = 0;
    candidates[1][val][row] = 0;
    candidates[2][val][col] = 0;
    candidates[3][val][box] = 0;

    for(int nval=0;nval<SIZE;nval++) {
        if( candidates[1][ nval ][ row ] == (1<<col) ) err = 1;
        if( candidates[2][ nval ][ col ] == (1<<row) ) err = 1;
        if( candidates[3][ nval ][ box ] == (1<<boxi) ) err = 1;

        candidates[1][ nval ][ row ] &= ~(1<<col);
        candidates[2][ nval ][ col ] &= ~(1<<row);
        candidates[3][ nval ][ box ] &= ~(1<<boxi);
    }
    for(int i=0;i<NEIGHBORS;i++) {
        int nidx = neighbors[idx][i];
        int nrow = nidx / SIZE;
        int ncol = nidx % SIZE;
        int nbox = (nrow / BOX_SIZE) * BOX_SIZE + (ncol / BOX_SIZE);
        int nboxi = (nrow % BOX_SIZE) * BOX_SIZE + (ncol % BOX_SIZE);

        if( candidates[0][nrow ][ ncol ] == (1<<val) ) err = 1;
        if( candidates[1][ val ][ nrow ] == (1<<ncol) ) err = 1;
        if( candidates[2][ val ][ ncol ] == (1<<nrow) ) err = 1;
        if( candidates[3][ val ][ nbox ] == (1<<nboxi) ) err = 1;

        candidates[0][nrow ][ ncol ] &= ~(1<<val);
        candidates[1][ val ][ nrow ] &= ~(1<<ncol);
        candidates[2][ val ][ ncol ] &= ~(1<<nrow);
        candidates[3][ val ][ nbox ] &= ~(1<<nboxi);
    }
    if(err) is_invalid = 1;
    return to_position(idx)*10 + val+1;
}


int find_next_moves(int *current) {
    for (int idx = 0; idx < NUM_CELLS; idx++) current[idx] = board[idx];
    int changes = 0;
    for (int g=0;g<GROUPS;g++) {
        for (int a=0;a<SIZE;a++) {
            for (int b=0;b<SIZE;b++) {
                if (__builtin_popcount(candidates[g][a][b]) == 1) {
                    int c = __builtin_ffs(candidates[g][a][b]) - 1;
                    int idx = gindex(g,a,b,c);
                    int val = gvalue(g,a,b,c);
                    if(current[idx]) {current[idx] += 10;continue;}
                    changes++;
                    current[idx] = 10 + val;
                }
            }
        }
    }
    return changes;
}


int apply_rules_all() {
    int current[NUM_CELLS];
    int changes = find_next_moves(current);
    if(changes == 0) return 0;
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        if(current[idx] < 10) continue;
        fill_value(idx, current[idx]%10);
    }
    return changes;
}

int new_board[NUM_CELLS];
int new_candidates[GROUPS][SIZE][SIZE];
int find_next_guess(int *current) {
    int changes = find_next_moves(current);
    if(changes) return changes;
    for(int p=0;p<NUM_CELLS;p++) {
        if(board[p]) continue;
        for(int val=0;val<SIZE;val++) {
            if(candidates[0][p / SIZE][p % SIZE] & (1<<val)) {
                memcpy(new_board, board, sizeof(board));
                memcpy(new_candidates, candidates, sizeof(candidates));

                fill_value(p,val);
                while(apply_rules_all());

                int ok = checkvalid();

                //reset
                memcpy(board, new_board, sizeof(board));
                memcpy(candidates, new_candidates, sizeof(candidates));
                is_invalid = 0;

                if(ok) {
                    changes++;
                    current[p] = 10 + val;
                    break;
                }
            }
        }
    }
    return changes;
}

int random_move(int *current) {
    int changes = 0;
    int res = -1;
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        if(current[idx] < 10) continue;
        changes++;
        if( rand() % changes == 0 ) res = idx;
    }
    return res >= 0 ? fill_value(res, current[res]%10) : 0;
}

int random_guess(int only_candidates) {
    int changes = 0;
    int res = -1, rval;

    for(int p=0;p<NUM_CELLS;p++) {
        if(board[p]) continue;
        for(int val=0;val<SIZE;val++) {
            if(only_candidates && !((candidates[0][p / SIZE][p % SIZE] >> val) & 1) ) continue;

            changes++;
            if( rand() % changes == 0 ) {
                res = p;
                rval = val;
            }
        }
    }
    return res >= 0 ? fill_value(res, rval) : 0;
}

int path[100];
int* solve_path(char *puzzle, char *labels) {

    init_neighbors();
    initialize();

    int cnt = -1;

    for (int i = 0; i < NUM_CELLS; i++) {
        if(puzzle[i] >= '1' && puzzle[i] <= '9') {
            path[++cnt] = fill_value(i,puzzle[i]-'0'-1);
        }
    }
    path[++cnt] = 100;
    int current[NUM_CELLS];
    while(!is_invalid && find_next_moves(current)) {
        for (int idx = 0; idx < NUM_CELLS; idx++) {
            if(current[idx] < 10) continue;
            int i = to_position(idx);
            labels[ cnt*100 + i] = '0'+current[idx]%10+1;
        }
        path[++cnt] = random_move(current);
    }

    if(is_invalid) {
        labels[cnt*100+10] = '2';
        path[++cnt] = 102;

        labels[cnt*100] = '0';
        path[++cnt] = 0;
        return path;
    }


    if(checkvalid()) {
        path[++cnt] = 0;
        return path;
    }

    labels[cnt*100+10] = '1';
    path[++cnt] = 101;

    find_next_guess(current);
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        if(current[idx] < 10) continue;
        int i = to_position(idx);
        labels[ cnt*100 + i] = '0'+current[idx]%10+1;
    }
    if(rand() % 3 == 0)
        path[++cnt] = random_move(current);
    else
        path[++cnt] = random_guess(rand() % 2);

    while(!is_invalid && find_next_moves(current)) {
        for (int idx = 0; idx < NUM_CELLS; idx++) {
            if(current[idx] < 10) continue;
            int i = to_position(idx);
            labels[ cnt*100 + i] = '0'+current[idx]%10+1;
        }
        path[++cnt] = random_move(current);
    }

    if(!checkvalid()) { //Checks both if there is conflict (is_invalid) AND if it is not filled fully
        labels[cnt*100+10] = '2';
        path[++cnt] = 102;
    }

    labels[cnt*100] = '0';
    path[++cnt] = 0;

    return path;
}

