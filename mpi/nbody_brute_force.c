/*
** nbody_brute_force.c - nbody simulation using the brute-force algorithm (O(n*n))
**
**/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <unistd.h>
#include <mpi.h>

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

#include "ui.h"
#include "nbody.h"
#include "nbody_tools.h"

FILE *f_out = NULL;

int nparticles = 10;      /* number of particles */
float T_FINAL = 1.0;     /* simulation end time */
particle_t *particles;

double sum_speed_sq = 0;
double max_acc = 0;
double max_speed = 0;

void init() {
    /* Nothing to do */
}

#ifdef DISPLAY
Display *theDisplay;  /* These three variables are required to open the */
GC theGC;             /* particle plotting window.  They are externally */
Window theMain;       /* declared in ui.h but are also required here.   */
#endif

/* compute the force that a particle with position (x_pos, y_pos) and mass 'mass'
 * applies to particle p
 */
void compute_force(particle_t *p, double x_pos, double y_pos, double mass) {
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
void move_particle(particle_t *p, double step) {

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
void all_move_particles(double step) {
    /* First calculate force for particles. */
    int i;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype MPI_PARTICLE_T;
    int blocklengths[7] = {1, 1, 1, 1, 1, 1, 1}; // Nombre de répétitions pour chaque champ
    MPI_Aint displacements[7];
    MPI_Datatype types[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}; // Types de données dans 'particle_t'

    displacements[0] = offsetof(particle_t, x_pos);
    displacements[1] = offsetof(particle_t, y_pos);
    displacements[2] = offsetof(particle_t, x_vel);
    displacements[3] = offsetof(particle_t, y_vel);
    displacements[4] = offsetof(particle_t, x_force);
    displacements[5] = offsetof(particle_t, y_force);
    displacements[6] = offsetof(particle_t, mass);

    // Créez le type MPI pour 'particle_t'.
    MPI_Type_create_struct(7, blocklengths, displacements, types, &MPI_PARTICLE_T);
    MPI_Type_commit(&MPI_PARTICLE_T);

    int nparticles_per_process = nparticles/size;

    for (i = 0; i < nparticles; i++) {

        particle_t* local_particles = malloc(sizeof(particle_t) * nparticles_per_process);

        MPI_Scatter(particles, nparticles_per_process, MPI_PARTICLE_T, local_particles, nparticles_per_process, MPI_PARTICLE_T, 0, MPI_COMM_WORLD);

        particle_t particule_i;
        particule_i.x_force = 0;
        particule_i.y_force = 0;
        int j;
        for (j = 0; j < nparticles_per_process; j++) {
            particle_t *p = &particles[j];
            /* compute the force of particle j on particle i */
            compute_force(&particule_i, p->x_pos, p->y_pos, p->mass);
        }

        if (rank == 0) {
            particle_t* all_particule_i = malloc(sizeof(particle_t) * nparticles_per_process);

            MPI_Gather(&particule_i, 1, MPI_PARTICLE_T, all_particule_i, 1, MPI_PARTICLE_T, 0, MPI_COMM_WORLD);

            particles[i].x_force = 0;
            particles[i].y_force = 0;
            for (int k = 0; k < size; k++) {
                particles[i].x_force += all_particule_i[i].x_force;
                particles[i].y_force += all_particule_i[i].y_force;
            }

        }
        else {
            MPI_Gather(&particule_i, 1, MPI_PARTICLE_T, NULL, 0, MPI_DATATYPE_NULL, 0, MPI_COMM_WORLD);
        }
    }

    /* then move all particles and return statistics */
    for (i = 0; i < nparticles; i++) {
        move_particle(&particles[i], step);
    }


    for (i = 0; i < nparticles; i++) {
        int j;
        particles[i].x_force = 0;
        particles[i].y_force = 0;

        printf("i: %d", i);

        for (j = nparticles * rank / size; j < nparticles * (rank + 1) / size; j++) {
            particle_t *p = &particles[j];
            /* compute the force of particle j on particle i */
            compute_force(&particles[i], p->x_pos, p->y_pos, p->mass);
        }

        MPI_Send(&particles[i].x_force, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&particles[i].y_force, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

        if (rank == 0) {

            for (int t = 1; t < size; t++) {
                MPI_Status status;
                double force;

                MPI_Recv(&force, 1, MPI_DOUBLE, t, 0, MPI_COMM_WORLD, &status);
                particles[i].x_force += force;

                MPI_Recv(&force, 1, MPI_DOUBLE, t, 1, MPI_COMM_WORLD, &status);
                particles[i].y_force += force;
            }

        }
    }

    /* then move all particles and return statistics */
    for (i = 0; i < nparticles; i++) {
        move_particle(&particles[i], step);
    }
}

/* display all the particles */
void draw_all_particles() {
    int i;
    for (i = 0; i < nparticles; i++) {
        int x = POS_TO_SCREEN(particles[i].x_pos);
        int y = POS_TO_SCREEN(particles[i].y_pos);
        draw_point(x, y);
    }
}

void print_all_particles(FILE *f) {
    int i;
    for (i = 0; i < nparticles; i++) {
        particle_t *p = &particles[i];
        fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
    }
}

void run_simulation() {
    double t = 0.0, dt = 0.01;

    while (t < T_FINAL && nparticles > 0) {
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
        clear_display();
        draw_all_particles();
        flush_display();
#endif
    }
}

/*
  Simulate the movement of nparticles particles.
*/
int main(int argc, char **argv) {
    if (argc >= 2) {
        nparticles = atoi(argv[1]);
    }
    if (argc == 3) {
        T_FINAL = atof(argv[2]);
    }

    init();

    /* Allocate global shared arrays for the particles data set. */
    particles = malloc(sizeof(particle_t) * nparticles);
    all_init_particles(nparticles, particles);

    /* Initialize thread data structures */
#ifdef DISPLAY
    /* Open an X window to display the particles */
    simple_init (100,100,DISPLAY_SIZE, DISPLAY_SIZE);
#endif

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    MPI_Init(&argc, &argv);
    /* Main thread starts simulation ... */
    run_simulation();
    MPI_Finalize();

    gettimeofday(&t2, NULL);

    double duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

#ifdef DUMP_RESULT
    FILE* f_out = fopen("particles.log", "w");
    assert(f_out);
    print_all_particles(f_out);
    fclose(f_out);
#endif

    printf("-----------------------------\n");
    printf("nparticles: %d\n", nparticles);
    printf("T_FINAL: %f\n", T_FINAL);
    printf("-----------------------------\n");
    printf("Simulation took %lf s to complete\n", duration);

#ifdef DISPLAY
    clear_display();
    draw_all_particles();
    flush_display();

    printf("Hit return to close the window.");

    getchar();
    /* Close the X window used to display the particles */
    XCloseDisplay(theDisplay);
#endif
    return 0;
}
