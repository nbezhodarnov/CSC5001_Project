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

#include <mpi.h>

#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <pthread.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <stdbool.h>

int nparticles = 10; /* number of particles */
int nparticles_at_start = 10;
float T_FINAL = 1.0; /* simulation end time */
particle_t *particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

node_t *root;

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

  /* recvcounts - number of bytes to receive from each node
   * displs - byte displacement of each node reply in the receive buffer
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

void init_tools(int *argc, char **argv)
{
  init_MPI(argc, argv);
}

void insert_all_particles(int nparticles, particle_t *particles, node_t *root);

void init(int *argc, char ***argv)
{
  init_tools(argc, argv);

  parse_args(*argc, *argv);

  if (*argc >= 4) {
    omp_set_num_threads(atoi((*argv)[3]));
  }

  nparticles_at_start = nparticles;
  
  init_alloc(8 * nparticles);
  root = malloc(sizeof(node_t));
  init_node(root, NULL, XMIN, XMAX, YMIN, YMAX);

  /* Allocate global shared arrays for the particles data set. */
  particles = malloc(sizeof(particle_t) * nparticles);

  if (rank == root_mpi_node) {
    all_init_particles(nparticles, particles);
  }

  MPI_Bcast(particles, nparticles, mpi_particle_type, root_mpi_node, MPI_COMM_WORLD);

  insert_all_particles(nparticles, particles, root);

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

/* compute the force that node n acts on particle p */
void compute_force_on_particle(node_t *n, particle_t *p)
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
    compute_force(p, n->x_center, n->y_center, n->mass);
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
      compute_force(p, n->x_center, n->y_center, n->mass);
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
        compute_force_on_particle(&n->children[i], p);
      }
    }
  }
}

bool is_particle_out_from_area(particle_t* p, node_t* root)
{
  return p->x_pos < root->x_min ||
      p->x_pos > root->x_max ||
      p->y_pos < root->y_min ||
      p->y_pos > root->y_max;
}

void compute_force_for_all_particles_in_root_area()
{
  #pragma omp parallel for schedule(dynamic)
  for (int i = index_start_local; i < index_end_local; i++)
  {
    particle_t* p = &particles[i];

    if (is_particle_out_from_area(p, root))
    {
      continue;
    }

    p->x_force = 0;
    p->y_force = 0;

    compute_force_on_particle(root, p);
  }
}

void compute_force_in_node(node_t *n)
{
  if (!n)
    return;

  if (n->particle)
  {
    particle_t *p = n->particle;
    p->x_force = 0;
    p->y_force = 0;
    compute_force_on_particle(root, p);
  }
  if (n->children)
  {
    int i;
    for (i = 0; i < 4; i++)
    {
      compute_force_in_node(&n->children[i]);
    }
  }
}

/* compute the new position/velocity */
void move_particle_without_insert_to_tree(particle_t *p, double step)
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

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);

  p->node = NULL;
}

/* compute the new position/velocity */
void move_particle(particle_t *p, double step, node_t *new_root)
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

  sum_speed_sq += speed_sq;
  max_acc = MAX(max_acc, cur_acc);
  max_speed = MAX(max_speed, cur_speed);

  p->node = NULL;
  if (p->x_pos < new_root->x_min ||
      p->x_pos > new_root->x_max ||
      p->y_pos < new_root->y_min ||
      p->y_pos > new_root->y_max)
  {
    nparticles--;
  }
  else
  {
    insert_particle(p, new_root);
  }
}

void move_all_particles_in_root_area(double step)
{
  #pragma omp for schedule(dynamic) reduction(+: sum_speed_sq) reduction(max: max_acc, max_speed)
  for (int i = index_start_local; i < index_end_local; i++)
  {
    particle_t* p = &particles[i];

    if (is_particle_out_from_area(p, root))
    {
      continue;
    }

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

      p->node = NULL;
    }
  }
}

/* compute the new position of the particles in a node */
void move_particles_in_node(node_t *n, double step, node_t *new_root)
{
  if (!n)
    return;

  if (n->particle)
  {
    particle_t *p = n->particle;
    move_particle(p, step, new_root);
  }
  if (n->children)
  {
    int i;
    for (i = 0; i < 4; i++)
    {
      move_particles_in_node(&n->children[i], step, new_root);
    }
  }
}

/*
  Move particles one time step.

  Update positions, velocity, and acceleration.
  Return local computations.
*/
void all_move_particles(double step)
{
  /* First calculate force for particles. */
  compute_force_for_all_particles_in_root_area();

  /* then move all particles and return statistics */
  move_all_particles_in_root_area(step);

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

  node_t *new_root = alloc_node();
  init_node(new_root, NULL, XMIN, XMAX, YMIN, YMAX);

  insert_all_particles(nparticles, particles, new_root);

  free_node(root);
  root = new_root;
}

#if DISPLAY
void draw_particles()
{
  draw_node(root);
}
#endif

#ifdef DUMP_RESULT
void dump_particles(FILE *f)
{
  print_particles(f, root);
}
#endif

void run_simulation()
{
  double t = 0.0, dt = 0.01;

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

    node_t *n = root;
    clear_display();
    draw_node(n);
    flush_display();
#endif
  }
}

/* create a quad-tree from an array of particles */
void insert_all_particles(int nparticles, particle_t *particles, node_t *root)
{
  int i;
  for (i = 0; i < nparticles; i++)
  {
    particle_t* p = &particles[i];

    if (is_particle_out_from_area(p, root))
    {
      continue;
    }

    insert_particle(p, root);
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

  free_node(root);
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
