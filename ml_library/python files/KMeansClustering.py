import numpy as np
import pandas as pd

class KMeansClustering:
    def __init__(self):
        self.x = None
        self.idx = None
        self.centroids = None
        self.k = None
        self.epochs_limit = None
        self.epochs_count = None

    def find_nearest_clusters(self):
        distances = np.linalg.norm(self.x[:, np.newaxis] - self.centroids, axis=2)
        self.idx = np.argmin(distances, axis=1)

    def compute_centroids(self):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            points_in_cluster = self.x[self.idx == i]
            if len(points_in_cluster) > 0:
                new_centroids[i] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[i] = self.x[np.random.choice(self.x.shape[0])]
        self.centroids = new_centroids

    def fit(self, x, k, epochs=50):
        self.x = x
        self.k = k
        self.epochs_limit = epochs
        self.epochs_count = 1

        indices = np.random.choice(self.x.shape[0], self.k, replace=False)
        self.centroids = self.x[indices].copy().astype(float)

        for _ in range(epochs):
            old_centroids = self.centroids.copy()
            self.find_nearest_clusters()
            self.compute_centroids()

            if np.allclose(old_centroids, self.centroids):
                break
            self.epochs_count += 1

    def calculate_inertia(self):
        inertia = 0
        for i in range(self.k):
            points = self.x[self.idx == i]
            if len(points) > 0:
                inertia += np.sum((points - self.centroids[i])**2)
        return inertia