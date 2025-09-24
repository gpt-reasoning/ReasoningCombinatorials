#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/sat_path.h"

int main(int argc, char *argv[]) {
    srand(0);
    DEBUGGING = 0;
    GeneratorParameters params = {25, 15, 1};

    //print_instance(&p);
    for(int i=0;i<10000;i++) {
        Problem p = generate_instance(&params);
        PathParameters pparams = {200, 0};
        solve_path(&p, &pparams);
    }
    /*
    for(int i=0;i<pparams.length;i++) {
        if(path[i] >= 100) printf("%d ", path[i]);
        else printf("%d ", token_var(path[i], &p));
        if(path[i] == 0) break;
    }
    printf("\n");
    */
    return 0;
}
