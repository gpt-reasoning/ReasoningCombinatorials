#ifndef SP_H
#define SP_H

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_PATH 2000
#define VOCAB_SIZE 2000

extern int path[MAX_PATH], pcnt;
extern char next_tokens[MAX_PATH][VOCAB_SIZE];

extern int DEBUGGING;
extern int NEGATIVETOKENS;
extern int DISPLAYLEVELS;
extern int FIRSTTOKEN;

void shuffle_array(int *A, int n);
void solve_path(char *puzzle);

#ifdef __cplusplus
}
#endif

#endif // SP_H