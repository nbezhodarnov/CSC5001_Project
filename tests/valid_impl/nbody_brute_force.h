#ifndef NBODY_BRUTE_FORCE_VALID
#define NBODY_BRUTE_FORCE_VALID

/*
** nbody_brute_force.h - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include "utils/nbody/nbody.h"

#include <stdio.h>

extern int nparticles; /* number of particles */
extern float T_FINAL; /* simulation end time */
extern particle_t *particles_valid;

extern double sum_speed_sq_valid;
extern double max_acc_valid;
extern double max_speed_valid;

void init_valid();

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force_valid(particle_t *p, double x_pos, double y_pos, double mass);

/* compute the new position/velocity */
void move_particle_valid(particle_t *p, double step);

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles_valid(double step);

void run_simulation_valid();

#endif
