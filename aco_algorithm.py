import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_cities = 20
num_ants = 40
num_iterations = 200
alpha = 1.0  #greater alpha means priority to amount of pheremone for choosing node
beta = 1.0  #greater beta means priority to distance between nodes
rho = 0.5  #decay rate 0.95 means small decay, 0.5 means larger decay
Q = 100 #Q is just a constant for calculating probability of ant choosing a ijth node

# Create random city coordinates
np.random.seed(0)
cities = np.random.rand(num_cities, 2) #generates random 2D coordinate for a city/node
#print(cities)
# cities=np.array([[0.5488135 , 0.71518937],
#                  [0.60276338, 0.54488318],
#                  [0.4236548 , 0.64589411],
#                  [0.43758721, 0.891773  ],
#                  [0.96366276 ,0.38344152],
#                  [0.79172504, 0.52889492],
#                  [0.56804456, 0.92559664],
#                  [0.07103606, 0.0871293 ],
#                  [0.0202184 , 0.83261985],
#                  [0.77815675 ,0.87001215],
#                  [0.97861834, 0.79915856],
#                  [0.46147936, 0.78052918],
#                  [0.11827443, 0.63992102],
#                  [0.14335329,0.94466892],
#                  [0.52184832,0.41466194],
#                  [0.26455561, 0.77423369],
#                  [0.45615033, 0.56843395],
#                  [0.0187898, 0.6176355 ],
#                  [0.61209572, 0.616934  ],
#                  [0.94374808, 0.6818203 ]])

# Calculate distance matrix
def calculate_distance_matrix(cities):
    num_cities = cities.shape[0]
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = np.linalg.norm(cities[i] - cities[j]) #basically it's distance formula
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

distance_matrix = calculate_distance_matrix(cities)
#print('distnace_matrix\n',distance_matrix)

# Initialize pheromone levels
pheromone_matrix = np.ones((num_cities, num_cities))

# Lists to store best solution history for visualization
best_distances = []
best_routes = []

# Ant Colony Optimization
best_solution = None
best_distance = float('inf')

for iteration in range(num_iterations):
    ant_routes = []
    ant_distances = []
    
    for ant in range(num_ants):
        #current_city = np.random.randint(num_cities)
        current_city = 18 #inital case we took 13th citiy
        unvisited_cities = set(range(num_cities))
        unvisited_cities.remove(current_city)
        route = [current_city]
        total_distance = 0
        
        while unvisited_cities:
            probabilities = (pheromone_matrix[current_city, list(unvisited_cities)] ** alpha) * \
                            ((1.0 / distance_matrix[current_city, list(unvisited_cities)]) ** beta)
            probabilities /= probabilities.sum()
            
            next_city = np.random.choice(list(unvisited_cities), p=probabilities)
            route.append(next_city)
            total_distance += distance_matrix[current_city, next_city]
            current_city = next_city
            unvisited_cities.remove(current_city)
        
        route.append(route[0])  # Return to the starting city
        total_distance += distance_matrix[current_city, route[0]]
        
        ant_routes.append(route)
        ant_distances.append(total_distance)
        
        if total_distance < best_distance:
            best_distance = total_distance
            best_solution = route
    
    # Pheromone update
    pheromone_matrix *= (1.0 - rho)
    for ant, route in enumerate(ant_routes):
        for i in range(num_cities):
            pheromone_matrix[route[i], route[i + 1]] += Q / ant_distances[ant]

     # Track best solution for visualization
    best_solution_index = np.argmin(ant_distances)
    best_distances.append(ant_distances[best_solution_index])
    best_routes.append(ant_routes[best_solution_index])

#Visualization of the best solution progression
plt.figure(figsize=(10, 6))
plt.plot(best_distances, marker='o', linestyle='-', color='b')
plt.xlabel('Iteration')
plt.ylabel('Best Distance')
plt.title(f'Best Distance Progression for alpha: {alpha}, beta: {beta}, rho: {rho}')
plt.grid(True)
plt.show(block=False)

#Visualization of the best solution
best_solution_cities = [cities[i] for i in best_solution]
best_solution_cities = np.array(best_solution_cities)

plt.figure(figsize=(10, 6))
plt.scatter(cities[:, 0], cities[:, 1], color='blue', label='Cities')
plt.plot(best_solution_cities[:, 0], best_solution_cities[:, 1], color='red', linewidth=2, label='Best Solution')
plt.scatter(best_solution_cities[0, 0], best_solution_cities[0, 1], color='green', label='Start City')
for i, city in enumerate(best_solution_cities):
    plt.text(city[0], city[1], str(i), fontsize=12, color='black')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title(f'TSP - Best Solution Distance: {best_distance:.2f} ,alpha: {alpha}, beta: {beta}, rho: {rho}')
plt.grid(True)
plt.show()
