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
#include <stdbool.h>

int nparticles = 10; /* number of particles */
float T_FINAL = 1.0; /* simulation end time */
particle_t *particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

extern bool display_enabled;

void init()
{
  /* Allocate global shared arrays for the particles data set. */
  particles = malloc(sizeof(particle_t) * nparticles);
  all_init_particles(nparticles, particles);
}

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
// #pragma omp declare simd
// void compute_force(double *x_force, double *y_force, double p_x_pos, double p_y_pos, double p_mass, 
//                    double x_pos, double y_pos, double mass)
// {
//   double x_sep, y_sep, dist_sq, grav_base;

//   x_sep = x_pos - p_x_pos;
//   y_sep = y_pos - p_y_pos;
//   dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

//   /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
//   grav_base = GRAV_CONSTANT * (p_mass) * (mass) / dist_sq;

//   (*x_force) += grav_base * x_sep;
//   (*y_force) += grav_base * y_sep;
// }

/* compute the new position/velocity */
// #pragma omp declare simd
// void move_particle(particle_t *p, double step)
// {
//   p->x_pos += (p->x_vel) * step;
//   p->y_pos += (p->y_vel) * step;
//   double x_acc = p->x_force / p->mass;
//   double y_acc = p->y_force / p->mass;
//   p->x_vel += x_acc * step;
//   p->y_vel += y_acc * step;

//   /* compute statistics */
//   double cur_acc = (x_acc * x_acc + y_acc * y_acc);
//   cur_acc = sqrt(cur_acc);
//   double speed_sq = (p->x_vel) * (p->x_vel) + (p->y_vel) * (p->y_vel);
//   double cur_speed = sqrt(speed_sq);

//   sum_speed_sq += speed_sq;
//   max_acc = MAX(max_acc, cur_acc);
//   max_speed = MAX(max_speed, cur_speed);
// }

// #pragma omp declare simd
// double simd_sqrt(const double val) {
//   return sqrt(val);
// }

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(double step)
{
  /* First calculate force for particles. */
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < nparticles; i++)
    {
      particle_t *p1 = &particles[i];
      double new_x_force = 0;
      double new_y_force = 0;

      #pragma omp simd reduction(+:new_x_force,new_y_force)
      for (int j = 0; j < nparticles; j++)
      {
        particle_t *p2 = &particles[j];

        // /* compute the force of particle j on particle i */
        double x_sep = p2->x_pos - p1->x_pos;
        double y_sep = p2->y_pos - p1->y_pos;
        double dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

        /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
        double grav_base = GRAV_CONSTANT * (p1->mass) * (p2->mass) / dist_sq;

        new_x_force += grav_base * x_sep;
        new_y_force += grav_base * y_sep;
      }

      p1->x_force = new_x_force;
      p1->y_force = new_y_force;
    }

    /* then move all particles and return statistics */
    #pragma omp for schedule(dynamic) reduction(+: sum_speed_sq) reduction(max: max_acc, max_speed)
    for (int i = 0; i < nparticles; i++)
    {
      particle_t *p = &particles[i];

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

      sum_speed_sq += speed_sq;
      max_acc = MAX(max_acc, cur_acc);
      max_speed = MAX(max_speed, cur_speed);
    }
  }
}

#if DISPLAY
void draw_particles()
{
  draw_all_particles();
}
#endif

#ifdef DUMP_RESULT
void dump_particles(FILE *f)
{
  print_all_particles(f);
}
#endif

void run_simulation()
{
  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles > 0)
  {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    dt = 0.1 * max_speed / max_acc;

    /* Plot the movement of the particle */
#if DISPLAY
    if (!display_enabled)
      continue;

    clear_display();
    draw_all_particles();
    flush_display();
#endif
  }
}
