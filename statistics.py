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
    print(data)
    create_graph("OpenMP", data, nb_particles, T_FINAL)


def mpi(n, openmp_n, nb_particles, T_FINAL):
    data = []
    for i in range(1, n + 1):
        print(f'Calculate for {i} different machines')
        data.append(get_duration(mpi_args(nb_particles, T_FINAL, i, openmp_n)))
    return data


def mpi_args(nb_particles, T_FINAL, i, openmp_n):
    return ['mpirun', '-np', str(i), '-hostfile', f'{FILEPATH}/mpi/hosts',
            f'{FILEPATH}/mpi/nbody_brute_force', str(nb_particles), str(T_FINAL), str(openmp_n)]


def mpi_graph(n, openmp_n, nb_particles, T_FINAL):
    data = mpi(n, openmp_n, nb_particles, T_FINAL)
    print(data)
    create_graph(f"MPI openMP: {openmp_n}", data, nb_particles, T_FINAL)


def create_graph(algo, data, nb_particles, T_FINAL):
    x = list(range(1, len(data) + 1))

    print(x)

    plt.plot(x, data, marker='o', linestyle='-')

    # Définissez les graduations personnalisées pour les axes x et y
    x_ticks = [i for i in range(1, n + 1)]
    # Utilisez xticks et yticks pour définir les graduations
    plt.xticks(x_ticks)

    plt.xlabel('Nb de processus')
    plt.ylabel('Temps (s)')
    plt.title(f'NBody Force Brute {algo} {nb_particles} {T_FINAL}')

    # plt.autoscale(False)
    plt.savefig(f'{FILEPATH}/graph/NBody Force Brute {algo} {nb_particles} {T_FINAL}.jpg')
    plt.clf()

def sequential_args(nb_particles, T_FINAL):
    return [f'{FILEPATH}/sequential/nbody_brute_force', str(nb_particles), str(T_FINAL)]


def bar_graph():
    title = "NBody Force Brute accélération des 3 méthodes"
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

        data_sequential.append((sequential_time/sequential_time) * 100)
        data_openmp.append((get_duration(openmp_args(nb_particles, T_FINAL, n))/sequential_time) * 100)
        data_mpi.append((get_duration(mpi_args(nb_particles, T_FINAL, n, 20))/sequential_time) * 100)
        data_cuda[i] = data_cuda[i]/sequential_time * 100

        legende.append(f"({nb_particles}, {T_FINAL})")

    # Créez un tableau d'indices pour les positions des barres
    indices = np.arange(len(legende))

    # Largeur des barres
    largeur_barre = 0.2

    # Créez le graphique en barres
    plt.bar(indices, data_sequential, width=largeur_barre, label='Sequential', color='blue')
    plt.bar(indices + largeur_barre, data_openmp, width=largeur_barre, label='OpenMP', color='green')
    plt.bar(indices + 2 * largeur_barre, data_mpi, width=largeur_barre, label='MPI', color='red')
    plt.bar(indices + 3 * largeur_barre, data_cuda, width=largeur_barre, label='CUDA', color='orange')

    # Étiquetez les axes et ajoutez une légende
    plt.xlabel('Paramètres')
    plt.ylabel('Accélération (%)')
    plt.title(title)
    plt.xticks(indices + largeur_barre*1.5, legende)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.subplots_adjust(right=0.82)

    # Affichez le graphique
    plt.savefig(f"{FILEPATH}/graph/{title}.jpg")
    plt.clf()


def evolution_t_graph(nb_particles):
    title = f"NBody Force Brute les 3 méthodes avec n_particle = {nb_particles}"

    data_sequential = []
    data_openmp = []
    data_mpi = []
    data_cuda = [2.297745, 4.279224, 6.238240, 8.506100, 10.356476, 12.732869, 14.744954, 16.276171, 18.448729, 21.032212]
    t_finals = range(1, 11)
    n = 5
    for t in t_finals:
        T_FINAL = t
        sequential_time = get_duration(sequential_args(nb_particles, T_FINAL))

        data_sequential.append(sequential_time)
        data_openmp.append(get_duration(openmp_args(nb_particles, T_FINAL, n)))
        data_mpi.append(get_duration(mpi_args(nb_particles, T_FINAL, n, 20)))
        data_cuda[t-1] = data_cuda[t-1]

    # Créez le graphique en barres
    plt.scatter(t_finals, data_sequential, label='Sequential', color='blue')
    plt.scatter(t_finals, data_openmp, label='OpenMP', color='green')
    plt.scatter(t_finals, data_mpi, label='MPI', color='red')
    plt.scatter(t_finals, data_cuda, label='CUDA', color='orange')

    # Étiquetez les axes et ajoutez une légende
    plt.xlabel('t')
    plt.ylabel('Temps (s)')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.subplots_adjust(right=0.82)

    # Affichez le graphique
    plt.savefig(f"{FILEPATH}/graph/{title}.jpg")
    plt.clf()


def evolution_nb_particles_graph(T_FINAL):
    title = f"NBody Force Brute les 3 méthodes avec T_FINAL = {T_FINAL}"

    data_sequential = []
    data_openmp = []
    data_mpi = []
    data_cuda = [4.671451, 13.238847, 24.246035, 37.037894, 49.437910]
    nb_particles_range = range(1000, 6000, 1000)
    n = 5
    for nb_particles in nb_particles_range:
        sequential_time = get_duration(sequential_args(nb_particles, T_FINAL))
        data_sequential.append(sequential_time)
        data_openmp.append(get_duration(openmp_args(nb_particles, T_FINAL, n)))
        data_mpi.append(get_duration(mpi_args(nb_particles, T_FINAL, n, 20)))
        data_cuda[(nb_particles//1000)-1] = data_cuda[(nb_particles//1000)-1]

    # Créez le graphique en barres
    plt.scatter(nb_particles_range, data_sequential, label='Sequential', color='blue')
    plt.scatter(nb_particles_range, data_openmp, label='OpenMP', color='green')
    plt.scatter(nb_particles_range, data_mpi, label='MPI', color='red')
    plt.scatter(nb_particles_range, data_cuda, label='CUDA', color='orange')

    # Étiquetez les axes et ajoutez une légende
    plt.xlabel('nb_particles')
    plt.ylabel('Temps (s)')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.subplots_adjust(right=0.82)

    # Affichez le graphique
    plt.savefig(f"{FILEPATH}/graph/{title}.jpg")
    plt.clf()


n = 10
number_particles = 5000
time = 2

mpi_graph(n, 1, number_particles, time)
mpi_graph(n, 20, number_particles, time)
openmp_graph(n, number_particles, time)
evolution_t_graph(1000)
evolution_nb_particles_graph(2)
bar_graph()
