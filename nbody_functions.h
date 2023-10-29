#ifndef NBODY_FUNCTIONS
#define NBODY_FUNCTIONS

/*
** nbody simulation functions header
**
**/

#include "utils/nbody/nbody.h"

#include <stdio.h>

extern int nparticles; /* number of particles */
extern float T_FINAL; /* simulation end time */

void init(int argc, char **argv);

#if DISPLAY
void draw_particles();
#endif

#ifdef DUMP_RESULT
void dump_particles(FILE *f);
#endif

void run_simulation();

#endif
