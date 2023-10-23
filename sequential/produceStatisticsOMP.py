import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

FILEPATH = os.path.dirname(os.path.abspath(__file__))

max_cpu = 20 # os.cpu_count()
data = []

nb_particles = 1000
T_FINAL = 3

for i in range(1, max_cpu + 1):
    if i > 1:
        print(f'Calculate for {i} thread')
    else:
        print(f'Calculate for {i} threads')

    result = subprocess.run([f'{FILEPATH}/nbody_brute_force', str(nb_particles), str(T_FINAL), str(i)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    print(output)

    lines = output.strip().split('\n')
    last_line = lines[-1]
    parts = last_line.split()
    if len(parts) > 0:
        time_taken = parts[2]
        time_taken = float(time_taken)
        data.append(time_taken)


x = list(range(1, max_cpu + 1))

plt.plot(x, data, marker='o', linestyle='-')
print(x)
print(data)

# Définissez les graduations personnalisées pour les axes x et y
x_ticks = [i for i in range(1, max_cpu + 1)]
y_ticks = [i for i in np.arange(0.5, 3, 0.3)]

# Utilisez xticks et yticks pour définir les graduations
plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.xlabel('Nb de threads')
plt.ylabel('Temps (s)')
plt.title('NBody Force Brute OpenMP 2')

plt.autoscale(False)
plt.show()
