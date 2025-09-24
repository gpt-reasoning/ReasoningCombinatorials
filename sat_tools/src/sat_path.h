#ifndef SP_H
#define SP_H

#ifdef __cplusplus
extern "C" {
#endif

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

void print_instance(Problem *problem) {
    printf("p cnf %d %d\n", problem->N, problem->M);
    for(int i = 0; i < problem->M; i++) {
        printf("%d %d %d 0\n", problem->clauses[i][0], problem->clauses[i][1], problem->clauses[i][2]);
    }
    fflush(stdout);
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


int merged[MAX_N];

int sign(int x) {return x < 0 ? -1 : (x > 0 ? 1 : 0);}

int find(int x) {
    if(merged[abs(x)] == 0) return x;
    merged[abs(x)] = find(merged[abs(x)]);
    return sign(x) * merged[abs(x)];
}


int merge(int x, int y, int sgn) {
    int px = find(x);
    int py = find(y);
    if (abs(px) == abs(py)) {
        if(px != sgn * py) return -1;
        return 0;
    }
    merged[abs(px)] = sgn * py * sign(px);
    return 1;
}


int findval(int x, int *filled) {
    int var = find(x);
    return sign(var) * filled[abs(var)];
}

int setval(int x, int *filled, int val) {
    int var = find(x);
    if(filled[abs(var)] ==  -val * sign(var)) return -1;
    filled[abs(var)] = val * sign(var);
    return 1;
}

int simplify_filled_true(PartialSolution *s, Problem *p, int *tokens) {
    int current[MAX_N];
    for(int i=1;i<=p->N;i++) current[i] = s->filled[i];
    for(int i = 0; i < p->M; i++) {
        for(int j = 0; j < 3; j++) {
            if (findval(p->clauses[i][j], s->filled) == +1) {
                for(int k=0; k<3; k++) {
                    if( k==j ) continue;
                    if(setval(p->clauses[i][k], current, -1) == -1) return -1;
                }
            }
        }
    }
    int n = 0;
    for(int i=1;i<=p->N;i++) {
        if (!s->filled[i] && findval(i, current)) tokens[n++] = var_token(i * findval(i, current), p);
    }
    return n;
}

int simplify_duplicates(PartialSolution *s, Problem *p, int *tokens) {
    int current[MAX_N];
    for(int i=1;i<=p->N;i++) current[i] = s->filled[i];

    for(int i = 0; i < p->M; i++) {
        for(int j = 0; j < 3; j++) {

            int var1 = find(p->clauses[i][(j+0)%3]);
            int var2 = find(p->clauses[i][(j+1)%3]);
            if (abs(var1) != abs(var2)) continue;
            int sign = var1 == var2 ? +1 : -1;
            if(setval(p->clauses[i][(j+2)%3], current, sign) == -1) return -1;
        }
    }
    
    int n = 0;
    for(int i=1;i<=p->N;i++) {
        if (!s->filled[i] && findval(i, current)) tokens[n++] = var_token(i * findval(i, current), p);
    }
    return n;
}

int simplify_filled_false(PartialSolution *s, Problem *p, int *tokens) {
    for(int i = 0; i < p->M; i++) {
        for(int j = 0; j < 3; j++) {
            if (findval(p->clauses[i][j], s->filled) == -1) {
                int r = merge(p->clauses[i][(j+1)%3], p->clauses[i][(j+2)%3], -1);
                if(r < 0) return -1;
            }
        }
    }
    return simplify_duplicates(s, p, tokens);
}


int logic_next_tokens(PartialSolution *s, Problem *p, int *tokens) {
    memset(merged, 0, sizeof(merged));
    int n = simplify_filled_true(s, p, tokens);
    if(n == -1) return -1;
    if(n > 0) return n;
    
    n = simplify_filled_false(s, p, tokens);
    if(n == -1) return -1;
    return n;
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
    int var = token_var(token, p);
    new_s.filled[abs(var)] = var / abs(var);
    return new_s;
}
int is_complete(PartialSolution *s, Problem *p) {
    for(int i = 0; i < p->M; i++) {
        int c = 0, all = 1;
        for(int j = 0; j < 3; j++) {
            int v = abs(p->clauses[i][j]);
            if(!s->filled[v]) {
                all = 0;
                continue;
            }
            c += p->clauses[i][j] == s->filled[v] * v;
        }
        if(c > 1) return -1;
        if(all && !c) return -1;
    }
    for(int i=1;i<=p->N;i++) {
        if(!s->filled[i]) return 0;
    }
    return 1;
}


#ifdef __cplusplus
}
#endif

#endif // SP_H