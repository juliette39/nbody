# Légende des graphs

## Open MP

![NBody Seq](NBody%20Seq.png)
graphe d'exécution de l'algorithme force brute en séquenceur

![NBody OpenMP 1](NBody%20OpenMP%201.png)

Graphe d'exécution de l'algorithme force brute avec Open MP 

> `#pragma omp parallel for default(none) shared(particles, nparticles)` ligne 97

![NBody OpenMP 2](NBody%20OpenMP%202.png)
Graphe d'exécution de l'algorithme force brute avec Open MP

> `#pragma omp parallel for default(none) shared(particles, nparticles)` ligne 102

![NBody OpenMP 3](NBody%20OpenMP%203.png)
Graphe d'exécution de l'algorithme force brute avec Open MP

> `#pragma omp parallel for default(none) shared(particles, nparticles)` ligne 97 & 102

## Conclusion
Le mieux est le mettre openMP à la ligne 97 (première boucle for)