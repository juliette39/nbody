import os
import subprocess
import matplotlib.pyplot as plt


max_cpu = os.cpu_count()
data = []

nb_particles = 5000
T_FINAL = 5

for i in range(1, max_cpu + 1) :
    print(f'Calculate for {i} core(s)')
    os.environ['OMP_NUM_THREADS'] = str(i)
    result = subprocess.run(['./nbody_barnes_hut', f'{nb_particles} {T_FINAL}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

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