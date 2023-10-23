# nbody

## Authors

- Théo Lardeur Gerzstein
- Juliette Debono

## OpenMP

```bash
make
```

## MPI

Run on differents machines

1. Connect to the machine
    ```bash
    ssh tsp # connect to a tsp machine
    ssh 3a401-05 # connect to a machine
    ```

2. Add the project and compile it

3. Test if it works

    ```bash
    wget https://www-inf.telecom-sudparis.eu/COURS/CSC5001/Supports/Cours/Intro/mpi_hello.c
    wget https://www-inf.telecom-sudparis.eu/COURS/CSC5001/Supports/Cours/Intro/hosts
    mpicc mpi_hello.c -o mpi_hello
    mpirun -np 13 -hostfile ./hosts ./mpi_hello
    ```

3. Run the project on all the machines
    ```bash
    mpirun -np 3 -hostfile ./hosts ./nbody_brute_force 1000 2 1 # run
    ```

## CUDA