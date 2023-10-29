/*
** main.c - main to launch nbody simulation
**
**/

#ifdef DISPLAY
#include "utils/ui/ui.h"
#endif

#include "nbody_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>

#ifdef DUMP_RESULT
#include <assert.h>
#endif

bool display_enabled = true;

/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char **argv)
{
  init(argc, argv);

  /* Initialize thread data structures */
#ifdef DISPLAY
  /* Open an X window to display the particles */
  simple_init(100, 100, DISPLAY_SIZE, DISPLAY_SIZE);
#endif

  struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  /* Main thread starts simulation ... */
  run_simulation();

  gettimeofday(&t2, NULL);

  double duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

#ifdef DUMP_RESULT
  FILE *f_out = fopen("particles.log", "w");
  assert(f_out);
  dump_particles(f_out);
  fclose(f_out);
#endif

  printf("-----------------------------\n");
  printf("nparticles: %d\n", nparticles);
  printf("T_FINAL: %f\n", T_FINAL);
  printf("-----------------------------\n");
  printf("Simulation took %lf s to complete\n", duration);

#ifdef DISPLAY
  clear_display();
  draw_particles();
  flush_display();

  printf("Hit return to close the window.");

  getchar();

  /* Close the X window used to display the particles */
  close_display();
#endif

  finalize();

  return 0;
}
