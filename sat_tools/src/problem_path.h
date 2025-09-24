#ifndef PP_H
#define PP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <assert.h>

int DEBUGGING = 0;

// Path constants
#define MAX_PATH 2000
#define VOCAB_SIZE 2000
#define PROMPT 100
#define ENDRULES 101
#define DEADEND 102
#define LEVEL 200

// Problem Constants
#define MAX_N 60
#define MAX_M 60

typedef struct Problem {
    int N, M;
    int clauses[MAX_M][3];
} Problem;

typedef struct GeneratorParameters {
    int N,M;
    int planted;
} GeneratorParameters;

typedef struct PartialSolution {
    int filled[MAX_PATH];
} PartialSolution;

typedef struct PathParameters {
    int length;
    int displaylevels;
    int firsttoken;
} PathParameters;

Problem generate_instance(GeneratorParameters *params);
int problem_tokens(Problem *p, int *tokens);
PartialSolution initial_solution(Problem *p);
int logic_next_tokens(PartialSolution *s, Problem *p, int *tokens);
int guess_next_tokens(PartialSolution *s, Problem *p, int *tokens);
int alternatives_next_tokens(int guess_token, PartialSolution *s, Problem *p, int *tokens);

PartialSolution apply_token(PartialSolution *s, Problem *p, int token);
int is_complete(PartialSolution *s, Problem *p);



// Helper function
void shuffle_array(int *A, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        int temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}



int path[MAX_PATH], pcnt;
char next_tokens[MAX_PATH][VOCAB_SIZE];

void add_token(int token, int level) {
    ++pcnt;
    if(pcnt >= MAX_PATH-1) {
        exit(0);
        if(DEBUGGING) {
            printf("MAX_PATH exceeded\n");
            fflush(stdout);
        }
        return;
    }
    path[pcnt] = token;
    if(!DEBUGGING) return;
    printf("\n");
    for(int t=0;t<level;t++) printf("  ");
    printf(" %02d:",token);
    fflush(stdout);
}

void add_next_token(int token) {
    if(pcnt >= MAX_PATH-1) {
        if(DEBUGGING) {
            printf("MAX_PATH exceeded\n");
            fflush(stdout);
        }
        return;
    }
    char *nt = next_tokens[pcnt];
    nt[ token ] = 1;
    if(!DEBUGGING) return;
    printf(" %02d", token);
    fflush(stdout);
}

void add_next_tokens(int *current, int n) {
    for(int i=0;i<n;i++) {
        add_next_token(current[i]);
    }
}

int recurse(int level, PartialSolution *s, Problem *p, PathParameters *params) {
    int token, n;
    int tokens[MAX_PATH];
    while(1) {
        n = logic_next_tokens(s, p, tokens);
        if(n < 0) {
            token = DEADEND;
            add_next_token(token);
            add_token(token, level);
            return 0;
        }
        if(n == 0) {
            token = ENDRULES;
            add_next_token(token);
            add_token(token, level);
            break;
        }
        add_next_tokens(tokens, n);
        token = tokens[ rand() % n ];
        add_token(token, level);
        *s = apply_token(s, p, token);
    }

    int complete = is_complete(s, p);
    if(complete > 0) return 1;
    if(complete < 0) return 0;

    if(params->displaylevels) {
        token = LEVEL+level;
        add_next_token(token);
        add_token(token, level);
    }
    
    n = guess_next_tokens(s, p, tokens);
    assert(n>0);
    add_next_tokens(tokens, n);
    token = tokens[ rand() % n ];
    n = alternatives_next_tokens(token, s, p, tokens);
    assert(n>0);

    shuffle_array(tokens, n);
    
    for(int i=0;i<n;i++) {
        token = tokens[i];
        if( i > 0 ) add_next_tokens(tokens + i, n - i);
        add_token(token, level);
        PartialSolution temp = apply_token(s, p, token);
        if(recurse(level+1, &temp, p, params)) return 1;

        if(params->displaylevels) {
            token = LEVEL+level;
            add_next_token(token);
            add_token(token, level);
        }
    }

    token = DEADEND;
    add_next_token(token);
    add_token(token, level);
    return 0;
}

int solve_path(struct Problem *p, struct PathParameters *params) {
    PartialSolution s = initial_solution(p);
    pcnt = -1;
    
    // Initialize next_tokens array
    memset(next_tokens, 0, sizeof(next_tokens));
    
    int tokens[MAX_PATH];
    int n = problem_tokens(p, tokens);
    for(int i=0;i<n;i++) add_token(tokens[i], 0);

    int token = PROMPT;
    add_token(token, 0);
    int v = recurse(1, &s, p, params);

    token = 0;
    add_next_token(token);
    add_token(token, 0);
    if(DEBUGGING){
        printf("\n");
        fflush(stdout);    
    }
    
    return v;
}

void initialize();


#ifdef __cplusplus
}
#endif

#endif // PP_H