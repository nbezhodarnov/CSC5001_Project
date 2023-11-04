/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/


#ifdef DISPLAY
#include "utils/ui/ui.h"
#endif

#include "utils/nbody/nbody.h"
#include "utils/nbody/nbody_tools.h"

#include <cuda.h>
#include <cuda_runtime.h>

int nparticles = 500; /* number of particles */
float T_FINAL = 1.0;  /* simulation end time */
particle_t *particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

#define MAX_THREADS 1024

// Custom realisations of atomicAdd and atomicMax for doubles if CUDA version < 6.0
// taken from
// https://stackoverflow.com/questions/17399119/cuda-atomicadd-for-double
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions%5B/url%5D

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

static __inline__ __device__ double atomicMax(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(max(val, __longlong_as_double(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

extern "C" void init(int argc, char **argv)
{
  parse_args(argc, argv);

  /* Allocate global shared arrays for the particles data set. */
  particles = (particle_t *)malloc(sizeof(particle_t) * nparticles);
  all_init_particles(nparticles, particles);
}

#if DISPLAY
void draw_particles()
{
  draw_all_particles();
}
#endif

#ifdef DUMP_RESULT

extern "C" void dump_particles(FILE *f)
{
  print_all_particles(f);
}
#endif

__global__ void reset_forces(particle_t *gpu_particles)
{
  int i = blockIdx.x;
  gpu_particles[i].x_force = 0;
  gpu_particles[i].y_force = 0;
}

__global__ void calculate_forces(particle_t *gpu_particles)
{
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // if j > nparticles, we don't want to do anything
  if (j >= gridDim.x)
    return;

  particle_t *p1 = &gpu_particles[i];
  particle_t *p2 = &gpu_particles[j];

  double x_sep, y_sep, dist_sq, grav_base;

  x_sep = p2->x_pos - p1->x_pos;
  y_sep = p2->y_pos - p1->y_pos;
  dist_sq = MAX((x_sep * x_sep) + (y_sep * y_sep), 0.01);

  /* Use the 2-dimensional gravity rule: F = d * (GMm/d^2) */
  grav_base = GRAV_CONSTANT * (p1->mass) * (p2->mass) / dist_sq;

  atomicAdd(&(p1->x_force), grav_base * x_sep);
  atomicAdd(&(p1->y_force), grav_base * y_sep);
}

__global__ void move_all_particles(particle_t *gpu_particles, double step, double *gpu_sum_speed_sq,
                                   double *gpu_max_acc, double *gpu_max_speed)
{
  const int i = blockIdx.x;

  particle_t *p = &gpu_particles[i];
  p->x_pos += (p->x_vel) * step;
  p->y_pos += (p->y_vel) * step;
  double x_acc = p->x_force / p->mass;
  double y_acc = p->y_force / p->mass;
  p->x_vel += x_acc * step;
  p->y_vel += y_acc * step;

  /* compute statistics */
  const double cur_acc = sqrt(x_acc * x_acc + y_acc * y_acc);
  const double speed_sq = (p->x_vel) * (p->x_vel) + (p->y_vel) * (p->y_vel);
  const double cur_speed = sqrt(speed_sq);

  atomicAdd(gpu_sum_speed_sq, speed_sq);
  atomicMax(gpu_max_acc, cur_acc);
  atomicMax(gpu_max_speed, cur_speed);
}

// Les kernel sont des points de synchro askip donc ca devrait etre bon
void all_move_particles_kernel(double step, particle_t *gpu_particles, double *gpu_sum_speed_sq,
                               double *gpu_max_acc, double *gpu_max_speed)
{
  reset_forces<<<nparticles, 1>>>(gpu_particles);

  // Since we can't have more than 1024 threads per block, we need to split the blocks
  const int blocks_count = max(1, int(ceil(double(nparticles) / double(MAX_THREADS))));

  calculate_forces<<<dim3(nparticles, blocks_count), dim3(1, MAX_THREADS)>>>(gpu_particles);
  move_all_particles<<<nparticles, 1>>>(gpu_particles, step, gpu_sum_speed_sq, gpu_max_acc, gpu_max_speed);
}

extern "C" void run_simulation()
{
  // CUDA setup
  particle_t *gpu_particles;
  double *gpu_sum_speed_sq, *gpu_max_acc, *gpu_max_speed;

  cudaMalloc((void **)&gpu_sum_speed_sq, sizeof(double));
  cudaMemcpy(&gpu_sum_speed_sq, &sum_speed_sq, sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&gpu_max_acc, sizeof(double));
  cudaMemcpy(&gpu_max_acc, &max_acc, sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&gpu_max_speed, sizeof(double));
  cudaMemcpy(&gpu_max_speed, &max_speed, sizeof(double), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&gpu_particles, nparticles * sizeof(particle_t));
  cudaMemcpy(gpu_particles, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);

  double t = 0.0, dt = 0.01;
  while (t < T_FINAL && nparticles > 0)
  {
    /* Update time. */
    t += dt;
    /* Move particles with the current and compute rms velocity. */
    all_move_particles_kernel(dt, gpu_particles, gpu_sum_speed_sq, gpu_max_acc, gpu_max_speed);

    /* Adjust dt based on maximum speed and acceleration--this
       simple rule tries to insure that no velocity will change
       by more than 10% */

    cudaMemcpy(&max_speed, gpu_max_speed, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_acc, gpu_max_acc, sizeof(double), cudaMemcpyDeviceToHost);

    // only for testing
    cudaMemcpy(&sum_speed_sq, gpu_sum_speed_sq, sizeof(double), cudaMemcpyDeviceToHost);

    dt = 0.1 * max_speed / max_acc;

    //printf("max_speed = %lf, max_acc = %lf, dt = %lf\n", max_speed, max_acc, dt);

    /* Plot the movement of the particle */
#if DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();
#endif
  }

  cudaMemcpy(particles, gpu_particles, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);
  cudaFree(gpu_particles);
  cudaFree(gpu_sum_speed_sq);
  cudaFree(gpu_max_acc);
  cudaFree(gpu_max_speed);

  cudaDeviceSynchronize();
}

// For compatibility with the other implementations
extern "C"
{
  void init_tools(int argc, char **argv) {}
  void finalize_tools() {}
  void finalize() {}
  void free_memory() {}
}