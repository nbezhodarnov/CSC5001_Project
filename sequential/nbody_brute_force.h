#ifndef NBODY_BRUTE_FORCE
#define NBODY_BRUTE_FORCE

/*
** nbody_brute_force.h - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include "utils/nbody/nbody.h"

#include <stdio.h>

extern int nparticles; /* number of particles */
extern float T_FINAL; /* simulation end time */
extern particle_t *particles;

extern double sum_speed_sq;
extern double max_acc;
extern double max_speed;

void init_tools(int *argc, char ***argv);

void init(int *argc, char ***argv);

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t *p, double x_pos, double y_pos, double mass);

/* compute the new position/velocity */
void move_particle(particle_t *p, double step);

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(double step);

#if DISPLAY
void draw_particles();
#endif

#ifdef DUMP_RESULT
void dump_particles(FILE *f);
#endif

void run_simulation();

void free_memory();

void finalize_tools();

void finalize();

#endif
