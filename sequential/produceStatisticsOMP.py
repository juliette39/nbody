import os
import subprocess
import matplotlib.pyplot as plt

FILEPATH = os.path.dirname(os.path.abspath(__file__))

max_cpu = os.cpu_count()
data = []

nb_particles = 2000
T_FINAL = 3

for i in range(1, max_cpu + 1):
    if i > 1:
        print(f'Calculate for {i} cores')
    else:
        print(f'Calculate for {i} core')

    result = subprocess.run([f'{FILEPATH}/nbody_brute_force', nb_particles, T_FINAL, i], stdout=subprocess.PIPE,
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
plt.show()