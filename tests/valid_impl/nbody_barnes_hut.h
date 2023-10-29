#ifndef NBODY_BARNES_HUT_VALID
#define NBODY_BARNES_HUT_VALID

/*
** nbody_barnes_hut.h - nbody simulation that implements the Barnes-Hut algorithm (O(nlog(n)))
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

extern node_t *root_valid;

void init_valid(int argc, char **argv);

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force_valid(particle_t *p, double x_pos, double y_pos, double mass);

/* compute the force that node n acts on particle p */
void compute_force_on_particle_valid(node_t *n, particle_t *p);

void compute_force_in_node_valid(node_t *n);

/* compute the new position/velocity */
void move_particle_valid(particle_t *p, double step, node_t *new_root);

/* compute the new position of the particles in a node */
void move_particles_in_node_valid(node_t *n, double step, node_t *new_root);

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles_valid(double step);

void run_simulation_valid();

/* create a quad-tree from an array of particles */
void insert_all_particles_valid(int nparticles, particle_t *particles, node_t *root);

void finalize_valid();

#endif
