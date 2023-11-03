/*
** nbody_barnes_hut.c - nbody simulation that implements the Barnes-Hut algorithm (O(nlog(n)))
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
int nparticles_valid = 10;
extern float T_FINAL; /* simulation end time */
particle_t *particles_valid;

double sum_speed_sq_valid = 0;
double max_acc_valid = 0;
double max_speed_valid = 0;

node_t *root_valid;

void insert_all_particles_valid(int particles_number, particle_t *particles, node_t *root);

void init_valid(int argc, char **argv)
{
  parse_args(argc, argv);
  nparticles_valid = nparticles;

  init_alloc(8 * nparticles_valid);
  root_valid = malloc(sizeof(node_t));
  init_node(root_valid, NULL, XMIN, XMAX, YMIN, YMAX);

  /* Allocate global shared arrays for the particles data set. */
  particles_valid = malloc(sizeof(particle_t) * nparticles_valid);
  all_init_particles(nparticles_valid, particles_valid);
  insert_all_particles_valid(nparticles_valid, particles_valid, root_valid);
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

/* compute the force that node n acts on particle p */
void compute_force_on_particle_valid(node_t *n, particle_t *p)
{
  if (!n || n->n_particles == 0)
  {
    return;
  }
  if (n->particle)
  {
    /* only one particle */
    assert(n->children == NULL);

    /*
      If the current node is an external node (and it is not body b),
      calculate the force exerted by the current node on b, and add
      this amount to b's net force.
    */
    compute_force_valid(p, n->x_center, n->y_center, n->mass);
  }
  else
  {
    /* There are multiple particles */

#define THRESHOLD 2
    double size = n->x_max - n->x_min; // width of n
    double diff_x = n->x_center - p->x_pos;
    double diff_y = n->y_center - p->y_pos;
    double distance = sqrt(diff_x * diff_x + diff_y * diff_y);

    /* Use the Barnes-Hut algorithm to get an approximation */
    if (size / distance < THRESHOLD)
    {
      /*
  The particle is far away. Use an approximation of the force
      */
      compute_force_valid(p, n->x_center, n->y_center, n->mass);
    }
    else
    {
      /*
        Otherwise, run the procedure recursively on each of the current
  node's children.
      */
      int i;
      for (i = 0; i < 4; i++)
      {
        compute_force_on_particle_valid(&n->children[i], p);
      }
    }
  }
}

void compute_force_in_node_valid(node_t *n)
{
  if (!n)
    return;

  if (n->particle)
  {
    particle_t *p = n->particle;
    p->x_force = 0;
    p->y_force = 0;
    compute_force_on_particle_valid(root_valid, p);
  }
  if (n->children)
  {
    int i;
    for (i = 0; i < 4; i++)
    {
      compute_force_in_node_valid(&n->children[i]);
    }
  }
}

/* compute the new position/velocity */
void move_particle_valid(particle_t *p, double step, node_t *new_root)
{
  assert(p->node != NULL);
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

  p->node = NULL;
  if (p->x_pos < new_root->x_min ||
      p->x_pos > new_root->x_max ||
      p->y_pos < new_root->y_min ||
      p->y_pos > new_root->y_max)
  {
    nparticles_valid--;
  }
  else
  {
    insert_particle(p, new_root);
  }
}

/* compute the new position of the particles in a node */
void move_particles_in_node_valid(node_t *n, double step, node_t *new_root)
{
  if (!n)
    return;

  if (n->particle)
  {
    particle_t *p = n->particle;
    move_particle_valid(p, step, new_root);
  }
  if (n->children)
  {
    int i;
    for (i = 0; i < 4; i++)
    {
      move_particles_in_node_valid(&n->children[i], step, new_root);
    }
  }
}

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles_valid(double step)
{
  /* First calculate force for particles. */
  compute_force_in_node_valid(root_valid);

  node_t *new_root = alloc_node();
  init_node(new_root, NULL, XMIN, XMAX, YMIN, YMAX);

  /* then move all particles and return statistics */
  move_particles_in_node_valid(root_valid, step, new_root);

  free_node(root_valid);
  root_valid = new_root;
}

void run_simulation_valid()
{
  double t = 0.0, dt = 0.01;

  while (t < T_FINAL && nparticles_valid > 0)
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

/* create a quad-tree from an array of particles */
void insert_all_particles_valid(int particles_number, particle_t *particles, node_t *root)
{
  int i;
  for (i = 0; i < particles_number; i++)
  {
    insert_particle(&particles[i], root);
  }
}

void free_memory_valid()
{
  free(particles_valid);
  free_node(root_valid);
}

void finalize_valid()
{
  free_memory_valid();
}
