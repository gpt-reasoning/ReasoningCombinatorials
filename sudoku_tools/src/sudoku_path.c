#include "sudoku_path.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#define SIZE 9
#define BOX_SIZE 3
#define GROUPS 4
#define NEIGHBORS 20
#define NUM_CELLS (SIZE * SIZE)


int path[MAX_PATH], pcnt;
char next_tokens[MAX_PATH][VOCAB_SIZE];

int DEBUGGING = 0;
int NEGATIVETOKENS = 0;
int DISPLAYLEVELS = 0;
int FIRSTTOKEN = 0;

int is_invalid;
int neighbors[NUM_CELLS][NEIGHBORS];
int board[NUM_CELLS];
int candidates[GROUPS][SIZE][SIZE];

int current[1000];

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
int to_token(int idx,int val) {return to_position(idx)*10+val+1;}
int token_idx(int token) {return (token/100-1)*9+token/10%10-1;}
int token_val(int token) {return token%10-1;}

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

int checkvalid() { // Checks for conflict
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

int update_candidates(int idx, int val) {
    int row = idx / SIZE;
    int col = idx % SIZE;
    int box = (row / BOX_SIZE) * BOX_SIZE + (col / BOX_SIZE);
    int boxi = (row % BOX_SIZE) * BOX_SIZE + (col % BOX_SIZE);
    int err = 0;

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
    if(err) {
        is_invalid = 1;
        return -1;
    }
    return to_token(idx,val);
}

int fill_value(int idx, int val) {
    board[idx] = val+1;
    return update_candidates(idx, val);
}


void calc_candidates() {
    for (int g=0;g<GROUPS;g++) {
        for (int a=0;a<SIZE;a++) {
            for (int b=0;b<SIZE;b++) {
                candidates[g][a][b] = (1<<SIZE)-1;
            }
        }
    }
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        if(board[idx]) {
            update_candidates(idx, board[idx]-1);
        }
    }
}


void initialize() {
    is_invalid = 0;
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        board[idx] = 0;
    }
    calc_candidates();

    memset(next_tokens, 0, sizeof(next_tokens));
}


int find_next_moves(int *current) {
    for (int idx = 0; idx < NUM_CELLS; idx++) current[idx] = 0;
    for (int g=0;g<GROUPS;g++) {
        for (int a=0;a<SIZE;a++) {
            for (int b=0;b<SIZE;b++) {
                if (__builtin_popcount(candidates[g][a][b]) == 1) {
                    int c = __builtin_ffs(candidates[g][a][b]) - 1;
                    int idx = gindex(g,a,b,c);
                    int val = gvalue(g,a,b,c);
                    current[idx] = to_token(idx,val);
                }
            }
        }
    }

    int changes = 0;
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        if(current[idx])
        current[changes++] = current[idx];
    }
    return changes;
}


int apply_rules_all() {
    int current[NUM_CELLS];
    int changes = find_next_moves(current);
    if(changes == 0) return 0;
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        if(!current[idx]) continue;
        fill_value(idx, current[idx]%10 - 1);
    }
    return changes;
}

int find_next_guess(int *current) {
    int mn = 10;
    for (int i = 0; i < NUM_CELLS; i++) {
        if(board[i]) {continue;}
        int v = __builtin_popcount(candidates[0][i / SIZE][i % SIZE]);
        if(v < mn) mn = v;
    }
    int tot = 0;
    for (int idx = 0; idx < NUM_CELLS; idx++) {
        if( __builtin_popcount(candidates[0][idx / SIZE][idx % SIZE]) != mn ) continue;
        
        for(int val=0;val<SIZE;val++) {
            if(!((candidates[0][idx / SIZE][idx % SIZE] >> val) & 1) ) continue;
            current[tot++] = to_token(idx,val);
        }
    }
    return tot;
}

void shuffle_array(int *A, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        int temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

}

int find_all_values(int idx, int *current) {
    int cnt = 0;
    for(int val=0;val<SIZE;val++) {
        if(!((candidates[0][idx / SIZE][idx % SIZE] >> val) & 1) ) continue;
        current[cnt++] = to_token(idx,val);
    }
    return cnt;
}



void add_token(int token, int level) {
    ++pcnt;
    if(pcnt >= MAX_PATH-1) {
        //printf("MAX_PATH exceeded\n");
        //fflush(stdout);
        return;
    }
    path[pcnt] = token;
    if(!DEBUGGING) return;
    printf("\n");
    for(int t=0;t<level;t++) printf("  ");

    if(token < 1000)
        printf(" %03d:",token);
    else
        printf("-%03d:", token % 1000);
    fflush(stdout);
}

void add_next_token(int token) {
    if(pcnt >= MAX_PATH-1) {
        //printf("MAX_PATH exceeded\n");
        //fflush(stdout);
        return;
    }
    char *nt = next_tokens[pcnt];
    nt[ token ] = 1;
    if(!DEBUGGING) return;
    if(token < 1000)
        printf(" %03d", token);
    else
        printf("-%03d", token% 1000);
    fflush(stdout);
}

void add_next_tokens(int *current, int n) {
    for(int i=0;i<n;i++) {
        add_next_token(current[i]);
    }
}

int recurse(int level) {
    int token;

    int moves[NUM_CELLS], allmoves = 0;

    while(!is_invalid) {
        int nmoves = find_next_moves(current);
        if( nmoves == 0) break;

        add_next_tokens(current, nmoves);

        token = FIRSTTOKEN ? current[0] : current[ rand() % nmoves ];
        fill_value(token_idx(token), token_val(token));
        add_token(token, level);
        moves[allmoves++] = token;
    }

    if(checkvalid()) {
        token = 0;
        add_next_token(token);
        add_token(token, 0);
        return 1;
    }

    if(!is_invalid) {
        
        token = 101;
        add_next_token(token);
        add_token(token, level);

        if(DISPLAYLEVELS) {
            token = level;
            add_next_token(token);
            add_token(token, level);
        }

        int total = find_next_guess(current);
        add_next_tokens(current, total);

        int vals[9];

        token = FIRSTTOKEN ? current[0] : current[rand() % total];
        int nvals = find_all_values(token_idx(token), vals);
        
        if(!FIRSTTOKEN) shuffle_array(vals, nvals);

        int _board[NUM_CELLS];
        for(int i=0;i<nvals;i++) {
            for(int j=0;j<NUM_CELLS;j++) _board[j] = board[j];
            
            token = vals[i];
            fill_value(token_idx(token), token_val(token));
            if(i>0) add_next_token(token);
            add_token(token, level);
            if( recurse(level+1) ) return 1;

            for(int j=0;j<NUM_CELLS;j++) board[j] = _board[j];
            calc_candidates();
            is_invalid = 0;

            if(NEGATIVETOKENS) {
                token = vals[i] + 1000;
                add_next_token(token);
                add_token(token, level);
            }
            
            if(DISPLAYLEVELS) {
                token = level;
                add_next_token(token);
                add_token(token, level);
            }
        }

    }


    token = 102;
    add_next_token(token);
    add_token(token, level);

    if(NEGATIVETOKENS) {
        while(allmoves>0) {
            token = moves[--allmoves] + 1000;
            add_next_token(token);
            add_token(token, level);
        }
    }
    return 0;
}

void solve_path(char *puzzle) {
    pcnt = -1;
    int token;
    init_neighbors();
    initialize();

    for (int i = 0; i < NUM_CELLS; i++) {
        if(puzzle[i] >= '1' && puzzle[i] <= '9') {
            token = fill_value(i,puzzle[i]-'0'-1);
            add_token(token, 0);
        }
    }
    token = 100;
    add_token(token, 0);
    recurse(1);

    if(!DEBUGGING) return;
    printf(" -%d-\n", pcnt);
    fflush(stdout);
}