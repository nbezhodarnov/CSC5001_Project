/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include "nbody_functions.h"

#ifdef DISPLAY
#include "utils/ui/ui.h"
#endif

#include "utils/nbody/nbody.h"
#include "utils/nbody/nbody_tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

extern int nparticles; /* number of particles */
extern float T_FINAL; /* simulation end time */
particle_t *particles_valid;

double sum_speed_sq_valid = 0;
double max_acc_valid = 0;
double max_speed_valid = 0;

void init_valid(int argc, char **argv)
{
  parse_args(argc, argv);

  /* Allocate global shared arrays for the particles data set. */
  particles_valid = malloc(sizeof(particle_t) * nparticles);
  all_init_particles(nparticles, particles_valid);
}

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force_valid(particle_t *p, double x_pos, double y_pos, double mass)
{
  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = x_pos - p->x_pos;
  y_sep = y_pos - p->y_pos;
  dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT * (p->mass) * (mass) / dist_sq;

  p->x_force += grav_base * x_sep;
  p->y_force += grav_base * y_sep;
}

/* compute the new position/velocity */
void move_particle_valid(particle_t *p, double step)
{
  p->x_pos += (p->x_vel) * step;
  p->y_pos += (p->y_vel) * step;
  double x_acc = p->x_force / p->mass;
  double y_acc = p->y_force / p->mass;
  p->x_vel += x_acc * step;
  p->y_vel += y_acc * step;

  /* compute statistics */
  double cur_acc = (x_acc * x_acc + y_acc * y_acc);
  cur_acc = sqrt(cur_acc);
  double speed_sq = (p->x_vel) * (p->x_vel) + (p->y_vel) * (p->y_vel);
  double cur_speed = sqrt(speed_sq);

  sum_speed_sq_valid += speed_sq;
  max_acc_valid = MAX(max_acc_valid, cur_acc);
  max_speed_valid = MAX(max_speed_valid, cur_speed);
}

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles_valid(double step)
{
  /* First calculate force for particles. */
  int i;
  for (i = 0; i < nparticles; i++)
  {
    int j;
    particles_valid[i].x_force = 0;
    particles_valid[i].y_force = 0;
    for (j = 0; j < nparticles; j++)
    {
      particle_t *p = &particles_valid[j];
      /* compute the force of particle j on particle i */
      compute_force_valid(&particles_valid[i], p->x_pos, p->y_pos, p->mass);
    }
  }

  /* then move all particles and return statistics */
  for (i = 0; i < nparticles; i++)
  {
    move_particle_valid(&particles_valid[i], step);
  }
}

void run_simulation_valid()
{
  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles > 0)
  {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles_valid(dt);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    dt = 0.1 * max_speed_valid / max_acc_valid;
  }
}
