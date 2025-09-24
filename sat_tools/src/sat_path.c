#include "sat_path.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

// Use the values from the header file
// #define MAX_PATH 400
// #define VOCAB_SIZE 1000

#define PROMPT 100
#define ENDRULES 101
#define DEADEND 102
#define LEVEL 200

#define MAX_N 50
#define MAX_M 100

// These are already declared in the header file as extern
int pcnt;

int PLANTED = 0;
int DEBUGGING = 0;
int NEGATIVETOKENS = 0;
int DISPLAYLEVELS = 0;
int FIRSTTOKEN = 0;

int path[MAX_PATH], pcnt;
char next_tokens[MAX_PATH][VOCAB_SIZE];

void shuffle_array(int *A, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        int temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}


int N, M;
int x[MAX_N];
int clauses[MAX_M][3];

void generate_assignment() {
    for(int i = 1; i <= N; i++) {
        x[i] = rand() % 2 ? +1 : -1;
        printf("%d ", x[i]);
    }
}
 
void generate_clauses() {
    for(int i = 0; i < M; i++) {
        int cnt = 0;
        do {
            cnt = 0;
            for(int j = 0; j < 3; j++) {
                clauses[i][j] = rand() % N + 1;
                int sign = (rand() % 2 == 0) ? 1 : -1;
                if (x[clauses[i][j]] == sign) cnt++;
                clauses[i][j] *= sign;
            }
        } while (
            abs(clauses[i][0]) == abs(clauses[i][1]) ||
            abs(clauses[i][0]) == abs(clauses[i][2]) ||
            abs(clauses[i][1]) == abs(clauses[i][2]) ||
            (cnt != 1 && PLANTED)
        );
    }
}


int filled[MAX_N];
int merged[MAX_N];

void initialize(int n, int m) {
    memset(filled, 0, sizeof(int)*MAX_N);
    memset(merged, 0, sizeof(int)*MAX_N);
    N = n;
    M = m;
    generate_assignment();
    generate_clauses();
}

int find(int x) {
    int s = x / abs(x);
    x = abs(x);
    if(merged[x] == 0) return s*x;
    merged[x] = find(merged[x]);
    return s*merged[x];
}


int findval(int x) {
    int s = x / abs(x);
    x = abs(x);
    int tok = find(x);
    int sign = tok / abs(tok);
    return s * filled[abs(tok)] * sign;
}

void merge(int x, int y, int sign) {
    int px = find(x);
    int py = sign * find(y);
    if (abs(px) == abs(py)) {
        //if(px != py) cout << "ERRORR"<< x << " " << y << " " << sign << endl;
        return;
    }
    printf("Merging %d %d %d\n", x, y, sign);
    int signx = (px / abs(px));
    merged[abs(px)] = py * signx;
}


int simplify_filled_true() {
    int ok = 0;
    int current[MAX_N];
    for(int i=1;i<=N;i++) current[i] = filled[i];
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < 3; j++) {
            if (findval(clauses[i][j]) == +1) {
                for(int k=0; k<3; k++) {
                    if( k==j ) continue;
                    int var2 = abs( find(clauses[i][k]) );
                    int sign2 = find(clauses[i][k]) / abs(find(clauses[i][k]));
                    if(!filled[var2]) ok = 1;
                    else if(filled[var2] != -sign2) return -1;
                    current[var2] = -sign2;
                }
            }
        }
    }
    for(int i=1;i<=N;i++) filled[i] = current[i];
    return ok;
}

int simplify_duplicates() {
    //cout<<"Calling simplify_duplicates"<<endl;
    int ok = 0;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < 3; j++) {
            if (abs( find(clauses[i][j]) ) == abs( find(clauses[i][(j+1)%3]) )) {
                int var = find(clauses[i][(j+2)%3]);
                int sign = var / abs(var);
                var = abs(var);
                if( find( clauses[i][j] ) != find( clauses[i][(j+1)%3] ) ) sign *= -1;
                
                if(!filled[var]) ok = 1;
                else if(filled[var] != sign) return -1;
                filled[var] = sign;
            }
        }
    }
    return ok;
}

int simplify_filled_false() {
    int ok = 0;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < 3; j++) {
            if (findval(clauses[i][j]) == -1) {
                int var1 = abs( find(clauses[i][(j+1)%3]) );
                int var2 = abs( find(clauses[i][(j+2)%3]) );
                int sign1 = find(clauses[i][(j+1)%3]) / var1;
                int sign2 = find(clauses[i][(j+2)%3]) / var2;
                if(var1 == var2) {
                    if(sign1 == sign2) return -1;
                    continue;
                }
                ok = 1;
                merge(var1, var2, -sign1*sign2);
            }
        }
    }
    if(ok) return simplify_duplicates();
    return 0;
}

void add_token(int token, int level) {
    if(token < 0) token = N - token;
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

    if(token > LEVEL)
        printf("L#%d:", token - LEVEL);
    else if(token <= N || token > 2*N)
        printf(" %02d:",token);
    else
        printf("-%02d:", (token - 1) % N + 1);
    fflush(stdout);
}

void add_next_token(int token) {
    if(token < 0) token = N - token;
    if(pcnt >= MAX_PATH-1) {
        //printf("MAX_PATH exceeded\n");
        //fflush(stdout);
        return;
    }
    char *nt = next_tokens[pcnt];
    nt[ token ] = 1;
    if(!DEBUGGING) return;
    
    if(token > LEVEL)
        printf(" L#%d", token - LEVEL);
    else if(token <= N || token > 2*N)
        printf(" %02d", token);
    else
        printf(" -%02d", (token - 1) % N + 1);
    fflush(stdout);
}

void add_next_tokens(int *current, int n) {
    for(int i=0;i<n;i++) {
        add_next_token(current[i]);
    }
}

int dfs(int level) {
    int token;
    int temp_filled[MAX_N];
    int temp_merged[MAX_N];

    int ok = 1;
    while(ok) {
        memcpy(temp_filled, filled, sizeof(int)*MAX_N);
        memcpy(temp_merged, merged, sizeof(int)*MAX_N);
        ok = 0;
        //cout<<"Calling simplify_filled_true"<<endl;
        ok = simplify_filled_true();
        if(ok < 0) {
            
            token = DEADEND;
            add_next_token(token);
            add_token(token, level);

            return 0;
        }
        if(ok) {
            int next_tokens[MAX_N] = {0};
            int next_ptr = 0;
            for(int i=1;i<=N;i++) {
                if(!temp_filled[ abs(find(i)) ] && filled[ abs(find(i)) ]) {
                    next_tokens[next_ptr++] = findval(i) * i;
                }
            }
            memcpy(filled, temp_filled, sizeof(int)*MAX_N);
            memcpy(merged, temp_merged, sizeof(int)*MAX_N);

            add_next_tokens(next_tokens, next_ptr);
            int token = next_tokens[ rand() % next_ptr ];
            add_token(token, level);
            filled[ abs(find(token)) ] = find(token) / abs(find(token));
            
            continue;
        }
        //cout<<"Calling simplify_filled_false"<<endl;
        ok = simplify_filled_false();
        if(ok < 0) {
            token = DEADEND;
            add_next_token(token);
            add_token(token, level);
            return 0;
        }
        if(ok) {
            int next_tokens[MAX_N] = {0};
            int next_ptr = 0;
            for(int i=1;i<=N;i++) {
                if(!temp_filled[ abs(find(i)) ] && filled[ abs(find(i)) ]) {
                    next_tokens[next_ptr++] = findval(i) * i;
                }
            }
            memcpy(filled, temp_filled, sizeof(int)*MAX_N);
            memcpy(merged, temp_merged, sizeof(int)*MAX_N);

            add_next_tokens(next_tokens, next_ptr);
            int token = next_tokens[ rand() % next_ptr ];
            add_token(token, level);
            filled[ abs(find(token)) ] = find(token) / abs(find(token));
            continue;
        }
    }



    int guess[2*N];
    int guess_ptr = 0;
    for(int i=1;i<=N;i++) {
        if(!filled[ abs(find(i)) ]) {
            guess[guess_ptr++] = i;
            guess[guess_ptr++] = -i;
        }
    }
    if(!guess_ptr) {
        return 1;
    }

    token = ENDRULES;
    add_next_token(token);
    add_token(token, level);

    if(DISPLAYLEVELS) {
        token = LEVEL+level;
        add_next_token(token);
        add_token(token, level);
    }
    
    add_next_tokens(guess, guess_ptr);


    token = FIRSTTOKEN ? guess[0] : guess[ rand() % guess_ptr ];
    
    add_token(token, level);

    int g = token;
    filled[ abs(find(token)) ] = find(token) / abs(find(token));
    if( dfs(level+1) ) return 1;
    memcpy(filled, temp_filled, sizeof(int)*MAX_N);
    memcpy(merged, temp_merged, sizeof(int)*MAX_N);
    token *= -1;
    
    if(DISPLAYLEVELS) {
        token = LEVEL+level;
        add_next_token(token);
        add_token(token, level);
    }


    token = g * -1;
    add_next_token(token);
    add_token(token, level);

    filled[ abs(find(token)) ] = find(token) / abs(find(token));
    
    if( dfs(level+1) ) return 1;
    memcpy(filled, temp_filled, sizeof(int)*MAX_N);
    memcpy(merged, temp_merged, sizeof(int)*MAX_N);

    if(DISPLAYLEVELS) {
        token = LEVEL+level;
        add_next_token(token);
        add_token(token, level);
    }
    
    token = DEADEND;
    add_next_token(token);
    add_token(token, level);
    return 0;
}


int solve_path(int N, int M, int planted) {
    PLANTED = planted;
    pcnt = -1;
    int token;
    initialize(N, M);

    for (int i = 0; i < M; i++) {
        for(int j=0;j<3;j++) {
            token = clauses[i][j];
            add_token(token, 0);
        }
    }
    token = PROMPT;
    add_token(token, 0);
    int res = dfs(1);


    token = 0;
    add_next_token(token);
    add_token(token, 0);

    if(!DEBUGGING) return res;
    printf(" -%d-\n", pcnt);
    fflush(stdout);
    return res;
}