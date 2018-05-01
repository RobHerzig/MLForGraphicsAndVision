#NAME: ROBERT HERZIG
#MATNO: 3605172
#COMPILED WITH PYTHON 3.6.5

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

num_of_random_points = 2**10

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

def exhaustive_search_with_queries(points, queries):
    start_time = datetime.datetime.utcnow()
    for q in queries:
        find_nearest_neighbour_exhaustively(points, q)

    end_time = datetime.datetime.utcnow()
    time_diff = end_time - start_time
    return time_diff.total_seconds() * 1000

def get_tree_from_array(array):
    tree = KDTree(array, leaf_size=2)
    return tree

def kdtree_search(tree, queries):
    start_time = datetime.datetime.utcnow()
    for q in queries:
        dist, ind = tree.query([q], k=3)
        # print(str(dist) + " " + str(ind))

    end_time = datetime.datetime.utcnow()
    time_diff = end_time - start_time
    return time_diff.total_seconds() * 1000

test_no_kd = True
test_kd = True

times_kd = []
times_no_kd = []

if True:
    # timings for different D
    if test_no_kd:
        for D in range(1, 500, 10):
            random_points = np.random.rand(num_of_random_points, D)
            time = exhaustive_search_with_queries(random_points, random_points) #just in case we want to use other points
            times_no_kd.append(time)
            print(str(time) + " FOR " + str(D) + " DIMENSIONS")

        plt.plot(range(1, 500, 10), times_no_kd)
        plt.title('Query Times without KD')
        plt.xlabel('dimension (D)')
        plt.ylabel('time (ms)')
        plt.savefig('1_1_a_NO_KD_TREE.png', bbox_inches='tight')

    if test_kd:
        # kd-tree search
        for D in range(1, 500, 10):
            print(D)
            random_points = np.random.rand(num_of_random_points, D)
            tree = get_tree_from_array(random_points)
            time = kdtree_search(tree, random_points)
            times_kd.append(time)
            print(str(time) + " FOR " + str(D) + " DIMENSIONS")

        # plot to file
        plt.clf()
        plt.plot(range(1, 500, 10), times_kd)
        plt.title('Query Times')
        plt.xlabel('dimension (D)')
        plt.ylabel('time (ms)')
        plt.savefig('1_1_c_ONLY_KD_TREE.png', bbox_inches='tight')

    if test_kd and test_no_kd:
        plt.clf()
        plt.plot(range(1, 500, 10), times_no_kd)
        plt.plot(range(1, 500, 10), times_kd)
        plt.title('Query Times')
        plt.xlabel('dimension (D)')
        plt.ylabel('time (ms)')
        plt.savefig('1_1_COMBINED.png', bbox_inches='tight')

