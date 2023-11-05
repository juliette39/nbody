
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

#ifdef DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif


#include "nbody.h"
#include "nbody_tools.h"

#include "cuda_stuff.cuh"


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



/*
  Place particles in their initial positions.
*/
void all_init_particles(int num_particles, particle_t *particles) {
    int i;
    double total_particle = num_particles;

    for (i = 0; i < num_particles; i++) {
        particle_t *particle = &particles[i];
#if 0
        particle->x_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
        particle->y_pos = ((rand() % max_resolution)- (max_resolution/2))*2.0 / max_resolution;
        particle->x_vel = particle->y_pos;
        particle->y_vel = particle->x_pos;
        printf("%d,%d\n", particle->x_pos, particle->y_pos);

#else
        particle->x_pos = i * 2.0 / nparticles - 1.0;
        particle->y_pos = 0.0;
        particle->x_vel = 0.0;
        particle->y_vel = particle->x_pos;
#endif
        particle->mass = 1.0 + (num_particles + i) / total_particle;
        particle->node = NULL;

        //insert_particle(particle, root);
    }
}

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void computeForcesKernel(particle_t *particles, int numParticles, double grav) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numParticles && j < numParticles) {

        // Compute the force of particle j on particle i
        double x_sep, y_sep, dist_sq, grav_base;

        x_sep = particles[j].x_pos - particles[i].x_pos;
        y_sep = particles[j].y_pos - particles[i].y_pos;
        dist_sq = fmax(x_sep * x_sep + y_sep * y_sep, 0.01);

        // Use the 2-dimensional gravity rule: F = G * (m1 * m2) / d^2
        grav_base = grav * particles[i].mass * particles[j].mass / dist_sq;

        //particles[i].x_force += grav_base * x_sep;
        //particles[i].y_force += grav_base * y_sep;


        atomicAddDouble(&(particles[i].x_force), grav_base * x_sep);
        atomicAddDouble(&(particles[i].y_force), grav_base * y_sep);
    }

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



void print_all_particles(FILE *f) {
    int i;
    for (i = 0; i < nparticles; i++) {
        particle_t *p = &particles[i];
        fprintf(f, "particle={pos=(%f,%f), vel=(%f,%f)}\n", p->x_pos, p->y_pos, p->x_vel, p->y_vel);
    }
}


void run_simulation() {

    dim3 gridDim(nparticles, nparticles);
    dim3 blockDim(128, 1);

    particle_t *d_particles;
    cudaMalloc((void**)&d_particles, sizeof(particle_t) * nparticles);


    double grav = 0.01;
    double t = 0.0;
    double dt = 0.01;
    while (t < T_FINAL && nparticles > 0) {

        /* Update time. */
        t += dt;

        // Copy each particle individually to the device
        for (int i = 0; i < nparticles; i++) {
            particles[i].x_force = 0;
            particles[i].y_force = 0;
            cudaMemcpy(&d_particles[i], &particles[i], sizeof(particle_t), cudaMemcpyHostToDevice);
        }


        computeForcesKernel<<<gridDim, blockDim>>>(d_particles, nparticles, grav);
        cudaDeviceSynchronize();

        // Copy the results back to the host
        for (int i = 0; i < nparticles; i++) {
            cudaMemcpy(&particles[i], &d_particles[i], sizeof(particle_t), cudaMemcpyDeviceToHost);
        }



        //then move all particles and return statistics
        for (int i = 0; i < nparticles; i++) {
            move_particle(&particles[i], dt);
        }


         /* Adjust dt based on maximum speed and acceleration--this
           simple rule tries to insure that no velocity will change
           by more than 10% */
        dt = 0.1 * max_speed / max_acc;

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
    particles = (particle_t*)malloc(sizeof(particle_t) * nparticles);
    all_init_particles(nparticles, particles);


    struct timeval t1, t2;
    gettimeofday(&t1, NULL);


    /* Main thread starts simulation ... */
    run_simulation();

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

    return 0;
}