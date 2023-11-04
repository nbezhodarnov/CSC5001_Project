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

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
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

int rank;
int size;

int index_start_local;
int index_end_local;
int nparticles_local;

int *recvcounts;
int *displs;

int block_lengths[8] = {1, 1, 1, 1, 1, 1, 1, 1};
MPI_Datatype types[8] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_LONG};
MPI_Aint offsets[8];
MPI_Datatype mpi_particle_type;

const int root_mpi_node = 0;

void create_mpi_particle_type()
{
  offsets[0] = offsetof(particle_t, x_pos);
  offsets[1] = offsetof(particle_t, y_pos);
  offsets[2] = offsetof(particle_t, x_vel);
  offsets[3] = offsetof(particle_t, y_vel);
  offsets[4] = offsetof(particle_t, x_force);
  offsets[5] = offsetof(particle_t, y_force);
  offsets[6] = offsetof(particle_t, mass);
  offsets[7] = offsetof(particle_t, node);

  MPI_Type_create_struct(8, block_lengths, offsets, types, &mpi_particle_type);
  MPI_Type_commit(&mpi_particle_type);
}

void finalize_MPI();

void init_MPI(int *argc, char ***argv)
{
  int initialized = 0;
  MPI_Initialized(&initialized);

  if (initialized)
    return;

  MPI_Init(argc, argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  create_mpi_particle_type();
}

void init_local_variables(int rank, int size)
{
  int nparticles_per_node = round((double)nparticles / (double)size);
  int nparticles_left = nparticles - nparticles_per_node * (size - 1);

  index_start_local = rank * nparticles_per_node;
  index_end_local = (rank + 1) * nparticles_per_node;
  if (rank == size - 1)
  {
    index_end_local = nparticles;
  }
  nparticles_local = index_end_local - index_start_local;

  recvcounts = malloc(sizeof(int) * size);
  displs = malloc(sizeof(int) * size);

  /* recvcounts - number of particles to receive from each node
   * displs - particles displacement of each node reply in the receive buffer
   */
  int i;
  for (i = 0; i < size - 1; i++)
  {
    recvcounts[i] = nparticles_per_node;
    displs[i] = i * nparticles_per_node;
  }
  recvcounts[size - 1] = nparticles_left;
  displs[size - 1] = (size - 1) * nparticles_per_node;
}

void init_tools(int *argc, char ***argv)
{
  init_MPI(argc, argv);
}

void init(int *argc, char ***argv)
{
  init_tools(argc, argv);

  parse_args(*argc, *argv);

  /* Allocate global shared arrays for the particles data set. */
  particles = malloc(sizeof(particle_t) * nparticles);
  all_init_particles(nparticles, particles);

  init_local_variables(rank, size);
}

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t *p, double x_pos, double y_pos, double mass)
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
void move_particle(particle_t *p, double step)
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

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);
}

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(double step)
{
  /* First calculate force for particles. */
  int i;
  for (i = index_start_local; i < index_end_local; i++)
  {
    int j;
    particles[i].x_force = 0;
    particles[i].y_force = 0;
    for (j = 0; j < nparticles; j++)
    {
      particle_t *p = &particles[j];
      /* compute the force of particle j on particle i */
      compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
    }
  }

  /* then move all particles and return statistics */
  for (i = index_start_local; i < index_end_local; i++)
  {
    move_particle(&particles[i], step);
  }

  particle_t *particles_received = malloc(sizeof(particle_t) * nparticles);

  MPI_Allgatherv(&particles[index_start_local],
                 nparticles_local,
                 mpi_particle_type,
                 particles_received,
                 recvcounts,
                 displs,
                 mpi_particle_type,
                 MPI_COMM_WORLD);

  free(particles);
  particles = particles_received;
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

  MPI_Bcast(particles, nparticles, mpi_particle_type, root_mpi_node, MPI_COMM_WORLD);

  while (t < T_FINAL && nparticles > 0)
  {
    MPI_Bcast(&dt, 1, MPI_DOUBLE, root_mpi_node, MPI_COMM_WORLD);

    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles(dt);

    double sum_speed_sq_local = sum_speed_sq;
    double max_acc_local = max_acc;
    double max_speed_local = max_speed;

    MPI_Reduce(&sum_speed_sq_local, &sum_speed_sq, 1, MPI_DOUBLE, MPI_SUM, root_mpi_node, MPI_COMM_WORLD);
    MPI_Reduce(&max_acc_local, &max_acc, 1, MPI_DOUBLE, MPI_MAX, root_mpi_node, MPI_COMM_WORLD);
    MPI_Reduce(&max_speed_local, &max_speed, 1, MPI_DOUBLE, MPI_MAX, root_mpi_node, MPI_COMM_WORLD);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    if (rank == root_mpi_node)
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

void finalize_MPI()
{
  int finalized = 0;
  MPI_Finalized(&finalized);

  if (finalized)
    return;

  MPI_Finalize();
}

void free_memory()
{
  free(particles);

  free(recvcounts);
  free(displs);
}

void finalize_tools()
{
  finalize_MPI();
}

void finalize()
{
  finalize_tools();

  free_memory();
}
