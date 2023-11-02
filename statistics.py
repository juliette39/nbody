import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

FILEPATH = os.path.dirname(os.path.abspath(__file__))


def get_duration(args):
    result = subprocess.run(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    print(output)

    lines = output.strip().split('\n')
    last_line = lines[-1]
    parts = last_line.split()
    if len(parts) > 0:
        time_taken = parts[2]
        time_taken = float(time_taken)
        return time_taken
    else:
        print(f"error")
        return 0


def openmp(n, nb_particles, T_FINAL):
    data = []
    for i in range(1, n + 1):
        print(f'Calculate for {i} thread')
        data.append(get_duration(openmp_args(nb_particles, T_FINAL, i)))
    return data


def openmp_args(nb_particles, T_FINAL, i):
    return [f'{FILEPATH}/openmp/nbody_brute_force', str(nb_particles), str(T_FINAL), str(i)]


def openmp_graph(n, nb_particles, T_FINAL):
    data = openmp(n, nb_particles, T_FINAL)
    create_graph("OpenMP", data, nb_particles, T_FINAL)


def mpi(n, nb_particles, T_FINAL):
    data = []
    for i in range(1, n + 1):
        print(f'Calculate for {i} different machines')
        data.append(get_duration(mpi_args(nb_particles, T_FINAL, i)))
    return data


def mpi_args(nb_particles, T_FINAL, i):
    return ['mpirun', '-np', str(i), '-hostfile', f'{FILEPATH}/mpi/hosts',
            f'{FILEPATH}/mpi/nbody_brute_force', str(nb_particles), str(T_FINAL), str(i)]


def mpi_graph(n, nb_particles, T_FINAL):
    data = mpi(n, nb_particles, T_FINAL)
    create_graph("MPI", data, nb_particles, T_FINAL)


def create_graph(algo, data, nb_particles, T_FINAL):
    x = list(range(1, len(data) + 1))

    print(x)

    plt.plot(x, data, marker='o', linestyle='-')

    # Définissez les graduations personnalisées pour les axes x et y
    x_ticks = [i for i in range(1, n + 1)]

    # Utilisez xticks et yticks pour définir les graduations
    plt.xticks(x_ticks)

    plt.xlabel('Nb de threads')
    plt.ylabel('Temps (s)')
    plt.title(f'NBody Force Brute {algo} {nb_particles} {T_FINAL}')

    # plt.autoscale(False)
    plt.savefig(f'{FILEPATH}/graph/NBody Force Brute {algo} {nb_particles} {T_FINAL}.jpg')


def sequential_args(nb_particles, T_FINAL):
    return [f'{FILEPATH}/sequential/nbody_brute_force', str(nb_particles), str(T_FINAL)]


def total():
    values = [(1000, 2), (2000, 2), (1000, 5), (1000, 10), (5000, 1)]

    data_sequential = []
    data_openmp = []
    data_mpi = []
    data_cuda = [4.721920, 14.052495, 11.095877, 22.028955, 28.343530]
    legende = []
    n = 5
    for i in range(len(values)):
        nb_particles, T_FINAL = values[i]
        sequential_time = get_duration(sequential_args(nb_particles, T_FINAL))

        data_sequential.append(sequential_time/sequential_time)
        data_openmp.append(get_duration(openmp_args(nb_particles, T_FINAL, n))/sequential_time)
        data_mpi.append(get_duration(mpi_args(nb_particles, T_FINAL, n))/sequential_time)
        data_cuda[i] = data_cuda[i]/sequential_time

        legende.append(f"({nb_particles}, {T_FINAL})")

    # Créez un tableau d'indices pour les positions des barres
    indices = np.arange(len(legende))

    # Largeur des barres
    largeur_barre = 0.2

    # Créez le graphique en barres
    plt.bar(indices, data_sequential, width=largeur_barre, label='Sequential', color='blue')
    plt.bar(indices + largeur_barre, data_openmp, width=largeur_barre, label='OpenMP', color='green')
    plt.bar(indices + 2 * largeur_barre, data_mpi, width=largeur_barre, label='MPI', color='red')
    plt.bar(indices + 3 * largeur_barre, data_mpi, width=largeur_barre, label='CUDA', color='yellow')

    # Étiquetez les axes et ajoutez une légende
    plt.xlabel('Arguments')
    plt.ylabel('Temps')
    plt.title("NBody Force Brute accélération 3 méthodes")
    plt.xticks(indices + largeur_barre, legende)
    plt.legend()

    # Affichez le graphique
    plt.savefig(f"{FILEPATH}/graph/NBody Force Brute accélération 3 méthodes.jpg")
    plt.show()


n = 10
number_particles = 1000
time = 2

total()
