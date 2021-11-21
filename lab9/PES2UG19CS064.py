import numpy as np


class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        def find_distance(a, b):
            distance_squares = [pow((a[i] - b[i]), 2) for i in range(len(a))]
            return pow(sum(distance_squares), 0.5)

        result = []
        for point in data.tolist():
            distances_from_centroids = []
            for centroid in self.centroids.tolist():
                distance = find_distance(point, centroid)
                distances_from_centroids.append(distance)
            centroid_number = distances_from_centroids.index(min(distances_from_centroids))
            result.append(centroid_number)
        return np.array(result)

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
        Change self.centroids
        """
        count_dict = {i:0 for i in cluster_assgn}
        avg_dict = {key:[] for key, value in count_dict.items()}
        for idx, point in enumerate(data.tolist()):
            cluster = cluster_assgn[idx]
            count_dict[cluster] += 1
            cluster_total = avg_dict[cluster]
            if len(cluster_total) == 0:
                avg_dict[cluster] = point
            else:
                avg_dict[cluster] = [cluster_total[i] + point[i] for i in range(len(point))]
        
        sorted_cluster_list = list(set(cluster_assgn))
        sorted_cluster_list.sort()

        centroids = []
        for cluster in sorted_cluster_list:
            cluster_count = count_dict[cluster]
            cluster_sum = avg_dict[cluster]
            centroids.append([i / cluster_count for i in cluster_sum])

        self.centroids = np.array(centroids)

    def evaluate(self, data):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
        Returns:
            metric : (float.)
        """
        def find_distance(a, b):
            distance_squares = [pow((a[i] - b[i]), 2) for i in range(len(a))]
            return sum(distance_squares)

        # centroid_assigned = self.e_step(data)
        # result = 0
        # for i in range(len(data)):
        #     result += find_distance(data[i], self.centroids[centroid_assigned[i]])

        result = 0
        for point in data.tolist():
            for centroid in self.centroids.tolist():
                result += find_distance(point, centroid)
        return result