#NAME: ROBERT HERZIG
#MATNO: 3605172
#COMPILED WITH PYTHON 3.6.5

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

num_of_random_points = 2**10
num_dimensions = 2

def get_dim_vals():
    dim_vals = []
    i = 1
    while i <= 491:
        dim_vals.append(i)
        i += 10
    return dim_vals

def find_nearest_neighbour_exhaustively(array, root_point):
    start_time = datetime.datetime.utcnow()
    # if (index_to_search_from < 0) or (index_to_search_from > len(array)):
    #     print("INVALID POINT")
    #     return -1

    cur_min = array[0]
    cur_min_dist = np.inf
    root = root_point
    for i in range(0, len(array)):
        distance = np.linalg.norm(array[i]-root)
        if distance < cur_min_dist:
            cur_min = array[i]

    end_time = datetime.datetime.utcnow()
    time_diff = end_time - start_time
    # print(str(len(root)) + "DIMS -> " + str(time_diff) + " || MIN = " + str(cur_min))
    return cur_min

def test_dimensions(low, hi):
    for dims in range(low, hi+1): #+1 because I want it to include the high value, pls don't judge
        rands = np.random.rand(num_of_random_points, dims)

        nearest_neighbour_of_first_point = find_nearest_neighbour_exhaustively(rands, 0)
        print("NEAREST NEIGHBOUR OF " + str(rands[0]) + " IS " + str(nearest_neighbour_of_first_point))

def benchmark_dimensions():
    dim_vals = get_dim_vals()
    print(dim_vals)
    times = []

    # dim_vals = [1,2,3,4,5]
    for i in dim_vals:
        start_time = datetime.datetime.utcnow()
        random_points = np.random.rand(num_of_random_points, i)
        # print(random_points[0])
        for point in random_points:
            find_nearest_neighbour_exhaustively(random_points, point)
        end_time = datetime.datetime.utcnow()
        time_diff = end_time - start_time
        times.append(time_diff.total_seconds() * 1000)
        print(str(i) + " DIMENSIONS TOOK: " + str(time_diff))

    print(times)
    plt.plot(dim_vals, times)

    axes = plt.gca()
    # axes.set_xlim([0,xmax])
    axes.set_ylim([0, 10000])

    plt.xlabel('Dimensions')
    plt.ylabel('Time in ms for exh. search')
    plt.title('Time for loop with diff. dims')
    plt.grid(True)
    plt.savefig("Ex01_1.png")
    plt.show()

benchmark_dimensions()
