#ifndef JCZ_H
#define JCZ_H

#ifdef __cplusplus
extern "C" {
#endif

// Generates a puzzle by attempting to remove numbers based on solution uniqueness
extern void JCZGenerate(char *puzzle, int *perm);

// Solves the puzzle, returning number of solutions (up to 'limit')
// If 'solutionPtr' is not NULL and a solution is found, it's written there
extern int JCZSolver(const char *puzzle, char *solutionPtr, int limit);

#ifdef __cplusplus
}
#endif

#endif // JCZ_H