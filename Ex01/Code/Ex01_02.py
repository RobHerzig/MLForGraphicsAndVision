# NAME: ROBERT HERZIG
# MATNO: 3605172
# NAME: LARS-CHRISTIAN ACHAUER
# MATNO: 3594908
# COMPILED WITH PYTHON 3.6.5

import gzip
import numpy as np
from sklearn.neighbors import KDTree


def load_mnist(kind='train', each=1):
    # labels_path = os.path.join(path, '%s−labels−idx1−ubyte.gz' % kind)
    # images_path = os.path.join(path, '%s−images−idx3−ubyte.gz' % kind)

    # had to hardcode this, as join seemed to corrupt the path in some way that pycharm wouldn't accept
    # - didn't want to waste too much time on this
    if kind == 'train':
        labels_path = "../../downloads/train-labels-idx1-ubyte.gz"
        images_path = "../../downloads/train-images-idx3-ubyte.gz"
    elif kind == 'test':
        labels_path = "../../downloads/t10k-labels-idx1-ubyte.gz"
        images_path = "../../downloads/t10k-images-idx3-ubyte.gz"
    else:
        print("INVALID KIND")
        return

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    images = images[::each, :]
    labels = labels[::each]

    return images, labels


def get_tree_from_array(array):
    tree = KDTree(array, leaf_size=2)
    return tree


def kdtree_search(test_data, queries, k_val, test_labels, query_labels):
    tree = get_tree_from_array(test_data)
    num_correct = 0
    num_false = 0
    for i in range(0, len(queries)):
        # print(str(i) + "/" + str(len(queries)) + " k=" + str(k_val))
        dist, ind = tree.query([queries[i]], k=k_val) #TODO: exclude neighbourship between identical points
        found_correct = False
        # print("IND" + str(ind))

        #DATA FOR CALCULATING TOP-K ACCURACY
        for neighbour_index in ind[0]:
            if test_labels[neighbour_index] == query_labels[i]:
                found_correct = True
        if found_correct:
            num_correct += 1
        else:
            num_false += 1

        # print("NEAREST " + str(k) + " NEIGHBOURS: " + str(ind))

    accuracy = num_correct / (num_false + num_correct)

    print("TOP K ACCURACY k=" + str(k_val) + " : " + str(accuracy))
    return accuracy  # returns the 3 nearest neighbours' indices

def binary_classifier_prec_rec(test_data, queries, test_labels, query_labels, reference_label = 2):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    tree = get_tree_from_array(test_data)
    for i in range(0, len(queries)):
        print(str(i) + "/" + str(len(queries)) + " label = " + str(reference_label))
        dist, ind = tree.query([queries[i]], k=1)

        neighbour_label = test_labels[ind[0][0]] # TODO: Correct format
        # print("NEIGHBOUR LABEL " + str(neighbour_label))

        is_neighbour_reference_label = False
        if neighbour_label == reference_label:
            is_neighbour_reference_label = True

        actual_label = query_labels[i]
        is_actually_reference_label = False
        if actual_label == reference_label:
            is_actually_reference_label = True

        if is_neighbour_reference_label and is_actually_reference_label:
            true_positives += 1
        elif is_neighbour_reference_label and not is_actually_reference_label:
            false_positives += 1
        elif not is_neighbour_reference_label and is_actually_reference_label:
            false_negatives += 1
        elif not is_neighbour_reference_label and not is_actually_reference_label:
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall

def calculations_three_nearest_for_each_image(training_images, training_labels, test_images, test_labels):
    for k in range(1, 11):
        accuracy = kdtree_search(test_images, training_images, k, test_labels, training_labels)
        print("ACCURACY FOR k=" + str(k) + " -> " + str(accuracy))


def calculations_precision_recall(training_images, training_labels, test_images, test_labels):
    precision_2, recall_2 = binary_classifier_prec_rec(test_images, training_images,
                                                       test_labels, training_labels, reference_label=2)
    precision_6, recall_6 = binary_classifier_prec_rec(test_images, training_images,
                                                       test_labels, training_labels, reference_label=6)

    print("PRECISION FOR REFERENCE LABEL 2: " + str(precision_2))
    print("RECALL FOR REFERENCE LABEL 2: " + str(recall_2))

    print("PRECISION FOR REFERENCE LABEL 6: " + str(precision_6))
    print("RECALL FOR REFERENCE LABEL 6: " + str(recall_6))


kind_train = 'train'
kind_test = 'test'
each_value = 10
training_images, training_labels = load_mnist(kind=kind_train, each=each_value)
test_images, test_labels = load_mnist(kind=kind_test, each=each_value)

calculations_three_nearest_for_each_image(training_images, training_labels, test_images, test_labels)
calculations_precision_recall(training_images, training_labels, test_images, test_labels)

print("PROFIT")

