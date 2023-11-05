# nbody

Parallel application of the nbody project.

We only made the parallelization of the nbody_brute_force algorithm.

## Authors

- Théo Lardeur Gerzstein
- Juliette Debono

## Table of content

- [Sequential](#sequential): Project without any parallelization
- [OpenMP](#openmp): Project parallelized with OpenMP
- [MPI](#mpi): Project parallelized with MPI
- [CUDA](#cuda): Project parallelized with CUDA

## Sequential

- [Sequential](/sequential)

The given project without any parallelization.

## OpenMP

- [OpenMP](/openmp)

> To execute, add the Makefile with the flags `-fopenmp`, 
> compile the project and run `./nbody_brute_force 1000 2` for example

## MPI

- [MPI](/mpi)

### Execution

Run on different machines

1. Connect to the machine
    ```bash
    ssh tsp # connect to a tsp machine
    ssh 3a401-05 # connect to a machine
    ```

2. Test if the configurations of the machines are correct with a simple hello world

    ```bash
    wget https://www-inf.telecom-sudparis.eu/COURS/CSC5001/Supports/Cours/Intro/mpi_hello.c
    wget https://www-inf.telecom-sudparis.eu/COURS/CSC5001/Supports/Cours/Intro/hosts
    mpicc mpi_hello.c -o mpi_hello
    mpirun -np 13 -hostfile ./hosts ./mpi_hello
    ```

3. Add the project and the makefile (`CC=mpicc`) and compile it

4. Run the project on all the machines

   > Don't forget to add the `hosts` file

    ```bash
    mpirun -np 3 -hostfile ./hosts ./nbody_brute_force 1000 2 # run
    ```

## CUDA

The CUDA project was made on Google Collab to use the GPU:
[Google Collab](https://colab.research.google.com/drive/1OXNOQBkLIYI8RCZmmyzsYEgJJM78ik-Q?usp=sharing&fbclid=IwAR0KmpPjJk7ogdrR2y85BILEd5aCQXF5yQ0Bd9MmIYrsBFFcHEBPByCmREk#scrollTo=6-y1M5IF_EMC)