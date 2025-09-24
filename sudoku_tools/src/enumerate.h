#ifndef ENUMERATE_H
#define ENUMERATE_H

#ifdef __cplusplus
extern "C" {
#endif

// Generates a puzzle by attempting to remove numbers based on solution uniqueness
extern void generate(long long idx, int perm_id);

#ifdef __cplusplus
}
#endif

#endif // ENUMERATE_H