#include "problem_path.h"


Problem generate_instance(GeneratorParameters *params) {
    Problem p;
    p.N = params->N;
    p.M = params->M;

    int x[MAX_N];

    for(int i = 1; i <= p.N; i++) {
        x[i] = rand() % 2 ? +1 : -1;
    }

    for(int i = 0; i < p.M; i++) {
        int cnt = 0;
        do {
            cnt = 0;
            for(int j = 0; j < 3; j++) {
                p.clauses[i][j] = rand() % p.N + 1;
                int sign = (rand() % 2 == 0) ? 1 : -1;
                if (x[p.clauses[i][j]] == sign) cnt++;
                p.clauses[i][j] *= sign;
            }
        } while (
            abs(p.clauses[i][0]) == abs(p.clauses[i][1]) ||
            abs(p.clauses[i][0]) == abs(p.clauses[i][2]) ||
            abs(p.clauses[i][1]) == abs(p.clauses[i][2]) ||
            (cnt != 1 && params->planted)
        );
    }

    return p;
}

int var_token(int var, Problem *p) {
    if(var < 0) return  p->N - var;
    return var;
}
int token_var(int token, Problem *p) {
    if(token > p->N) return -(token-p->N);
    return token;
}

int problem_tokens(Problem *p, int *tokens) {
    int n = 0;
    for(int i = 0; i < p->M; i++) {
        for(int j = 0; j < 3; j++) {
            tokens[n++] = var_token(p->clauses[i][j], p);
        }
    }
    return n;
}
PartialSolution initial_solution(Problem *p) {
    PartialSolution s;
    memset(s.filled, 0, sizeof(s.filled));
    return s;
}

int logic_next_tokens(PartialSolution *s, Problem *p, int *tokens) {
    if(is_complete(s,p) < 0) return -1;
    return 0;
}
int guess_next_tokens(PartialSolution *s, Problem *p, int *tokens) {
    int n = 0;
    for(int i=1;i<=p->N;i++) if(!s->filled[i]) {
        tokens[n++] = var_token(i, p);
        tokens[n++] = var_token(-i, p);
    }
    return n;
}
int alternatives_next_tokens(int guess_token, PartialSolution *s, Problem *p, int *tokens) {
    tokens[0] = var_token( token_var(guess_token, p), p );
    tokens[1] = var_token( -token_var(guess_token, p), p );
    return 2;
}

PartialSolution apply_token(PartialSolution *s, Problem *p, int token) {
    PartialSolution new_s = *s;
    new_s.filled[abs(token)] = token / abs(token);
    return new_s;
}
int is_complete(PartialSolution *s, Problem *p) {
    for(int i = 0; i <= p->M; i++) {
        int c = 0, all = 1;
        for(int j = 0; j < 3; j++) {
            if(!s->filled[abs(p->clauses[i][j])]) {
                all = 0;
                continue;
            }
            c += (p->clauses[i][j] != s->filled[abs(p->clauses[i][j])]);
        }
        if(c > 1) return -1;
        if(all && !c) return -1;
    }
    for(int i=1;i<=p->N;i++) if(!s->filled[i]) return 0;
    return 1;
}
