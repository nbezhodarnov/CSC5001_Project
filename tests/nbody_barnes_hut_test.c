#define _GNU_SOURCE

#include "valid_impl/nbody_barnes_hut.h"
#include "sequential/nbody_barnes_hut.h"

#include "utils/nbody/nbody_tools.h"

#include <check.h>

#include <stdio.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

bool display_enabled = false;

void setup(void)
{
    // First argument is unused, second is nparticles, third is T_FINAL
    int argc = 3;
    char *argv[] = {"nbody_barnes_hut_test", "10", "3"};

    init(argc, argv);
    init_valid(argc, argv);
}

void teardown(void)
{
    free_memory();
    free_memory_valid();
}

#define ck_assert_double_eq_tolerant(X, Y) ck_assert_double_eq_tol(X, Y, 1e-6)

START_TEST(test_compute_force)
{
    int i;
    for (i = 0; i < nparticles; i++)
    {
        int j;
        particles[i].x_force = 0;
        particles[i].y_force = 0;
        particles_valid[i].x_force = 0;
        particles_valid[i].y_force = 0;
        for (j = 0; j < nparticles; j++)
        {
            particle_t *p = &particles[j];
            particle_t *p_valid = &particles_valid[j];

            compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
            compute_force_valid(&particles_valid[i], p_valid->x_pos, p_valid->y_pos, p_valid->mass);

            ck_assert_double_eq_tolerant(particles[i].x_force, particles_valid[i].x_force);
            ck_assert_double_eq_tolerant(particles[i].y_force, particles_valid[i].y_force);
        }
    }
}
END_TEST

START_TEST(test_compute_force_on_particle)
{
    int i;
    for (i = 0; i < nparticles; i++)
    {
        particles[i].x_force = 0;
        particles[i].y_force = 0;
        particles_valid[i].x_force = 0;
        particles_valid[i].y_force = 0;

        compute_force_on_particle(root, &particles[i]);
        compute_force_on_particle_valid(root_valid, &particles_valid[i]);

        ck_assert_double_eq_tolerant(particles[i].x_force, particles_valid[i].x_force);
        ck_assert_double_eq_tolerant(particles[i].y_force, particles_valid[i].y_force);
    }
}
END_TEST

START_TEST(test_compute_force_in_node)
{
    compute_force_in_node(root);
    compute_force_in_node_valid(root_valid);

    int i;
    for (i = 0; i < nparticles; i++)
    {
        ck_assert_double_eq_tolerant(particles[i].x_force, particles_valid[i].x_force);
        ck_assert_double_eq_tolerant(particles[i].y_force, particles_valid[i].y_force);
    }
}
END_TEST

START_TEST(test_all_move_particles)
{
    all_move_particles(0.01);
    all_move_particles_valid(0.01);

    int i;
    for (i = 0; i < nparticles; i++)
    {
        ck_assert_double_eq_tolerant(particles[i].x_force, particles_valid[i].x_force);
        ck_assert_double_eq_tolerant(particles[i].y_force, particles_valid[i].y_force);

        ck_assert_double_eq_tolerant(particles[i].x_pos, particles_valid[i].x_pos);
        ck_assert_double_eq_tolerant(particles[i].y_pos, particles_valid[i].y_pos);
        ck_assert_double_eq_tolerant(particles[i].x_vel, particles_valid[i].x_vel);
        ck_assert_double_eq_tolerant(particles[i].y_vel, particles_valid[i].y_vel);

        // These checks are commented to accept mpi optimizations
        // ck_assert_double_eq_tolerant(sum_speed_sq, sum_speed_sq_valid);
        // ck_assert_double_eq_tolerant(max_acc, max_acc_valid);
        // ck_assert_double_eq_tolerant(max_speed, max_speed_valid);
    }
}
END_TEST

START_TEST(test_run_simulation)
{
    printf("Starting simulation\n");
    run_simulation();
    printf("Starting valid simulation\n");
    run_simulation_valid();

    int i;
    for (i = 0; i < nparticles; i++)
    {
        ck_assert_double_eq_tolerant(particles[i].x_force, particles_valid[i].x_force);
        ck_assert_double_eq_tolerant(particles[i].y_force, particles_valid[i].y_force);

        ck_assert_double_eq_tolerant(particles[i].x_pos, particles_valid[i].x_pos);
        ck_assert_double_eq_tolerant(particles[i].y_pos, particles_valid[i].y_pos);
        ck_assert_double_eq_tolerant(particles[i].x_vel, particles_valid[i].x_vel);
        ck_assert_double_eq_tolerant(particles[i].y_vel, particles_valid[i].y_vel);

        // These checks are commented to accept mpi optimizations
        // ck_assert_double_eq_tolerant(sum_speed_sq, sum_speed_sq_valid);
        // ck_assert_double_eq_tolerant(max_acc, max_acc_valid);
        // ck_assert_double_eq_tolerant(max_speed, max_speed_valid);
    }
}
END_TEST

Suite *nbody_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("nbody barnes hut");

    tc_core = tcase_create("Core");
    tcase_add_checked_fixture(tc_core, setup, teardown);
    tcase_add_test(tc_core, test_compute_force);
    tcase_add_test(tc_core, test_compute_force_on_particle);
    tcase_add_test(tc_core, test_compute_force_in_node);
    tcase_add_test(tc_core, test_all_move_particles);
    tcase_add_test(tc_core, test_run_simulation);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(int argc, char **argv)
{
    init_tools(argc, argv);

    double number_failed;
    Suite *s;
    SRunner *sr;

    s = nbody_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_VERBOSE);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    finalize_tools();

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
