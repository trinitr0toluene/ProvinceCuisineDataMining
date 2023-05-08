import sys
import numpy as np





sys.path.append("../")


def euclidean_dist(vector_a, vector_b):
    """

    :param vector_a: a
    :param vector_b: b
    :return: euclidean distance between a and b
    """
    return np.sqrt(sum(np.power((vector_a - vector_b), 2)))


def generate_randCentroids(dataset, k):
    """

    :param dataset: data
    :param k: number of clusters
    :return: k centroids
    """
    num_features = dataset.shape[1]
    centroids = np.zeros((k, num_features))
    for j in range(num_features):
        min_value = min(dataset[:, j])
        value_range = float(max(dataset[:, j]) - min_value)
        centroids[:, j] = (min_value + value_range * np.random.rand(k)).reshape(k)

    return centroids


def exam_kmeans(dataset, k, iter_rounds, cluster_info=None, centroids=None):
    """

    :param dataset: data
    :param k: number of clusters
    :param iter_rounds: training rounds
    :param cluster_info: cluster class of data
    :param centroids: cluster centroids
    :return:
    """
    dataset_size = dataset.shape[0]
    cluster_dist = np.zeros(dataset_size)

    if cluster_info is None:
        cluster_info = np.zeros(dataset_size)
    if centroids is None:
        centroids = generate_randCentroids(dataset, k)
        print(centroids)

    for iter_index in range(iter_rounds):
        for data_index in range(dataset_size):
            min_dist = float('inf')
            min_index = -1
            for centroid_index in range(k):
                euc_dist = euclidean_dist(centroids[centroid_index],
                                          dataset[data_index])
                if euc_dist < min_dist:
                    min_dist = euc_dist
                    min_index = centroid_index
            # if cluster_info[data_index] != min_index:
            #     pass
            cluster_info[data_index], cluster_dist[data_index] = min_index, min_dist ** 2

        for centroid_index in range(k):
            data_belong_to_centroid_index = dataset[np.nonzero(cluster_info == centroid_index)[0]]
            print(data_belong_to_centroid_index)
            centroids[centroid_index] = np.mean(data_belong_to_centroid_index, axis=0)
            print(centroids)
            print("=======")

    return centroids, cluster_info, cluster_dist