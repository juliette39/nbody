import os
import subprocess
import matplotlib
matplotlib.use('Agg')
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
    data_sequentiel = [get_duration(sequential_args(nb_particles, T_FINAL))] * n
    print(data)
    print(data_sequentiel)
    x = list(range(1, len(data) + 1))

    print(x)

    plt.plot(x, data, marker='o', linestyle='-', label='OpenMP', color='green')
    plt.plot(x, data_sequentiel, marker='o', linestyle='-', label='Séquentiel', color='blue')

    # Définissez les graduations personnalisées pour les axes x et y
    x_ticks = [i for i in range(1, n + 1)]
    # Utilisez xticks et yticks pour définir les graduations
    plt.xticks(x_ticks)

    plt.xlabel('Nb de threads')
    plt.ylabel('Temps (s)')
    plt.title(f'NBody Force Brute OpenMP n = {nb_particles} T = {T_FINAL}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.subplots_adjust(right=0.82)

    # plt.autoscale(False)
    plt.savefig(f'{FILEPATH}/graph/NBody Force Brute OpenMP {nb_particles} {T_FINAL}.jpg')
    plt.clf()


def mpi(n, openmp_n, nb_particles, T_FINAL):
    data = []
    for i in range(1, n + 1):
        print(f'Calculate for {i} different machines')
        data.append(get_duration(mpi_args(nb_particles, T_FINAL, i, openmp_n)))
    return data


def mpi_args(nb_particles, T_FINAL, i, openmp_n):
    return ['mpirun', '-np', str(i), '-hostfile', f'{FILEPATH}/mpi/hosts_2',
            f'{FILEPATH}/mpi/nbody_brute_force', str(nb_particles), str(T_FINAL), str(openmp_n)]

def sequentiel(n, nb_particles, T_FINAL):
    data = []
    for i in range(n):
        data.append(get_duration(sequential_args(nb_particles, T_FINAL)))
    return data


def mpi_graph(n, openmp_n, nb_particles, T_FINAL):
    data_1 = mpi(n, 1, nb_particles, T_FINAL)
    data_n = mpi(n, openmp_n, nb_particles, T_FINAL)
    data_sequentiel = [get_duration(sequential_args(nb_particles, T_FINAL))] * n

    # data_1 = [19.169321, 10.651778, 8.201374, 6.771136, 6.328926, 5.752534, 5.530823, 5.086228, 5.747662, 5.451276]
    # data_n = [4.811567, 4.079158, 4.849799, 4.346784, 4.559994, 4.341067, 4.226174, 4.078246, 4.934475, 5.188528]
    # data_sequentiel = [18.739791, 18.739791, 18.739791, 18.739791, 18.739791, 18.739791, 18.739791, 18.739791, 18.739791, 18.739791]

    print(data_1)
    print(data_n)
    print(data_sequentiel)

    x = list(range(1, len(data_1) + 1))

    plt.plot(x, data_1, marker='o', linestyle='-', label='MPI sans OpenMP', color='red')
    plt.plot(x, data_n, marker='o', linestyle='-', label=f'MPI avec OpenMP\n{openmp_n} threads', color='purple')
    plt.plot(x, data_sequentiel, marker='o', linestyle='-', label='Séquentiel', color='blue')

    # Définissez les graduations personnalisées pour les axes x et y
    x_ticks = [i for i in range(1, n + 1)]
    # Utilisez xticks et yticks pour définir les graduations
    plt.xticks(x_ticks)

    plt.xlabel("Nombre de machines")
    plt.ylabel('Temps (s)')
    plt.title(f'NBody Force Brute MPI n = {nb_particles} T = {T_FINAL}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.subplots_adjust(right=0.75)

    # plt.autoscale(False)
    plt.savefig(f'{FILEPATH}/graph/NBody Force Brute MPI {nb_particles} {T_FINAL}.jpg')
    plt.clf()

def sequential_args(nb_particles, T_FINAL):
    return [f'{FILEPATH}/sequential/nbody_brute_force', str(nb_particles), str(T_FINAL)]


def bar_graph(coeur):
    title = "NBody Force Brute accélération des 3 méthodes"
    values = [(1000, 2), (2000, 2), (1000, 5), (1000, 10), (5000, 1)]

    data_sequential = []
    data_openmp = []
    data_mpi = []
    data_cuda = [4.178279, 12.758548, 9.293374, 18.270145, 23.828161]
    legende = []
    n = 5
    for i in range(len(values)):
        nb_particles, T_FINAL = values[i]
        sequential_time = get_duration(sequential_args(nb_particles, T_FINAL))

        data_sequential.append((sequential_time/sequential_time) * 100)
        data_openmp.append((get_duration(openmp_args(nb_particles, T_FINAL, n))/sequential_time) * 100)
        data_mpi.append((get_duration(mpi_args(nb_particles, T_FINAL, n, coeur))/sequential_time) * 100)
        data_cuda[i] = data_cuda[i]/sequential_time * 100

        legende.append(f"({nb_particles}, {T_FINAL})")

    # Créez un tableau d'indices pour les positions des barres
    indices = np.arange(len(legende))

    # Largeur des barres
    largeur_barre = 0.2

    # Créez le graphique en barres
    plt.bar(indices, data_sequential, width=largeur_barre, label='Séquentiel', color='blue')
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


def evolution_t_graph(nb_particles, coeur):
    title = f"NBody Force Brute les 3 méthodes avec n_particles = {nb_particles}"

    data_sequential = []
    data_openmp = []
    data_mpi = []
    data_cuda = [2.151075, 3.984043, 5.810516, 7.831068, 9.059130, 11.160149, 12.892105, 14.870889, 16.352423, 18.021320]

    t_finals = range(1, 11)
    n = 10
    for t in t_finals:
        T_FINAL = t
        sequential_time = get_duration(sequential_args(nb_particles, T_FINAL))

        data_sequential.append(sequential_time)
        data_openmp.append(get_duration(openmp_args(nb_particles, T_FINAL, coeur)))
        data_mpi.append(get_duration(mpi_args(nb_particles, T_FINAL, n, coeur)))
        data_cuda[t-1] = data_cuda[t-1]

    # Créez le graphique en barres
    plt.scatter(t_finals, data_sequential, label='Séquentiel', color='blue')
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


def evolution_nb_particles_graph(T_FINAL, coeur):
    title = f"NBody Force Brute les 3 méthodes avec T_FINAL = {T_FINAL}"

    data_sequential = []
    data_openmp = []
    data_mpi = []
    data_cuda = [4.117001, 12.752132, 22.066512, 32.404332, 46.290168]

    nb_particles_range = range(1000, 6000, 1000)
    n = 10
    for nb_particles in nb_particles_range:
        sequential_time = get_duration(sequential_args(nb_particles, T_FINAL))
        data_sequential.append(sequential_time)
        data_openmp.append(get_duration(openmp_args(nb_particles, T_FINAL, coeur)))
        data_mpi.append(get_duration(mpi_args(nb_particles, T_FINAL, n, coeur)))
        data_cuda[(nb_particles//1000)-1] = data_cuda[(nb_particles//1000)-1]

    # Créez le graphique en barres
    plt.scatter(nb_particles_range, data_sequential, label='Séquentiel', color='blue')
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


host = 10
coeur = os.cpu_count()
number_particles = 5000
time = 2

# mpi_graph(host, coeur, number_particles, time)
openmp_graph(coeur, number_particles, time)
# evolution_t_graph(1000, coeur)
# evolution_nb_particles_graph(2, coeur)
# bar_graph(coeur)
