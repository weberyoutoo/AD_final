# %%
########################################################
########  Do not modify the sample code segment ########
########################################################

import torchvision
import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import ticker

seed = 0
np.random.seed(seed)


def resample_total(data, label, ratio=0.05):
    """
        data: np.array, shape=(n_samples, n_features)
        label: np.array, shape=(n_samples,)
        ratio: float, ratio of samples to be selected
    """
    new_data = []
    new_label = []
    for i in range(10):
        i_data = data[label == i]
        idx = np.random.choice(list(range(len(i_data))),
                               int(len(i_data)*ratio))
        new_data.append(i_data[idx])
        new_label.append(np.ones(len(idx))*i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label


def resample(data, label, outlier_ratio=0.01, target_label=0):
    """
        data: np.array, shape=(n_samples, n_features)
        label: np.array, shape=(n_samples,)
        outlier_ratio: float, ratio of outliers
        target_label: int, the label to be treated as normal
    """
    new_data = []
    new_label = []
    for i in range(10):
        if i != target_label:
            i_data = data[label == i]
            target_size = len(data[label == target_label])
            num = target_size*((outlier_ratio/9))
            idx = np.random.choice(
                list(range(len(i_data))), int(num), replace=False)
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx))*i)
        else:
            new_data.append(data[label == i])
            new_label.append(np.ones(len(data[label == i]))*i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label


"""
====================================================================================================================
    The implementation of {KNN, K-Means, Distance-Base, LOF} algorithm by following.
"""


def euclidean(point, data, axis=None):
    return np.sqrt(np.sum((point - data)**2, axis=axis))


class KNN_AD:
    def __init__(self, k=1):
        '''
        Implement of Anomaly Detection Algorithm with KNN-classifier

        :param k: "K" neariset neighbor
        '''
        self.k = k

    def fit(self, X):
        self.X = X

    def score(self, Y):
        return pairwise_distances(X=Y, Y=self.X)

    def prediction(self, Y, normal=False):
        # Get sorted pairwise distances training and testing data
        pair_dist = self.score(Y)
        sorted_idx = np.argsort(pair_dist)
        sorted_idx = sorted_idx[:, :self.k]

        selected = [self.X[index] for index in sorted_idx]
        selected = np.array(selected)

        output = [np.sum(euclidean(point=Y[idx], data=selected[idx], axis=1))
                  for idx in range(Y.shape[0])]
        output = np.array(output)
        output = output / self.k

        # Mapping output -> [0,1]
        if normal:
            Min = min(output)
            Max = max(output)
            output = (output - Min)/(Max - Min)

        return output


class KMeans_AD:
    def __init__(self, n_clusters=1, max_iter=300):
        '''
        Implement of Anomaly Detection Algorithm with k-means

        :param n_clusters: number of clusters
        :param max_iter: number of maximum iteration
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = np.array([X[i] for i in idx])

        iteration = 0
        prev_centroids = np.array(None)
        while (self.centroids != prev_centroids).any() and iteration < self.max_iter:

            dists = pairwise_distances(X, self.centroids)
            centroids_id = np.argmin(dists, axis=1)

            buckets = []
            for i in range(self.n_clusters):
                buckets.append(np.where(centroids_id == i))

            prev_centroids = np.copy(self.centroids)
            for i in range(self.n_clusters):
                self.centroids[i] = np.mean(X[buckets[i]], axis=0)

            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def score(self, Y):
        return pairwise_distances(X=Y, Y=self.centroids)

    def prediction(self, Y, normal=False):
        pair_dists = self.score(Y)
        output = np.min(pair_dists, axis=1)

        if normal:
            Min = min(output)
            Max = max(output)
            output = (output - Min)/(Max - Min)

        return output


"""
    The implementaion of Distance Base AD algorithm and distance metics
"""


def CosineDistance(X, Y):
    X_dot_Y = np.dot(X, Y)
    X_norm = np.linalg.norm(X)
    Y_norm = np.linalg.norm(Y)
    return 1 - (X_dot_Y / (X_norm * Y_norm))


def L1Distance(X, Y):
    return np.sum(np.abs(X - Y))


def L2Distance(X, Y):
    return np.linalg.norm(X - Y)


def ChebyshevDistance(X, Y):
    return np.max(np.abs(X-Y))


def MahalanobisDistance(X, Y, VI):
    diff = X - Y
    return np.sqrt(np.matmul(np.matmul(diff, VI), diff.T))


class Distance_AD:
    def __init__(self, k=5):
        self.k = k
        self.metrics = {"CosineDistance": CosineDistance,
                        "L1Distance": L1Distance,
                        "L2Distance": L2Distance,
                        "ChebyshevDistance": ChebyshevDistance,
                        "MahalanobisDistance": MahalanobisDistance}

    def predict(self, X, ref=None, metric="L2Distance"):
        if (ref != None).any() and metric == "MahalanobisDistance":
            self.VI = np.linalg.inv(np.cov(ref.T))
            dists = pairwise_distances(
                X=X, metric=self.metrics[metric], VI=self.VI)
        else:
            dists = pairwise_distances(X=X, metric=self.metrics[metric])
        output = np.sort(dists, axis=1)[:, self.k]
        return output


class LOF:
    '''
        Implement of Density Base Anomaly Detection Algorithm
    '''

    def __init__(self, k=5):
        self.k = k

    def _KNN(self, X):
        dists = pairwise_distances(X=X)
        k_near = np.argsort(dists, axis=1)[:, 1:self.k+1]
        k_dist = np.sort(dists, axis=1)[:, self.k]

        return dists, k_near, k_dist

    def _reach_distance(self, dists, k_dist):
        k_dist_ex = np.expand_dims(k_dist, axis=0)
        k_dist_rep = np.repeat(k_dist_ex, self.size, axis=0)
        reach_dist = np.maximum(k_dist_rep, dists)
        assert (reach_dist > 0).all(
        ), "Error:: Reachable Distance expect to be > 0, but receive <= 0 value"

        return reach_dist

    def _LRD(self, reach_dist, k_near):
        # Local reachablility distance (lrd)
        k_reachDist = [reach_dist[idx, k_near[idx]]
                       for idx in range(self.size)]
        lrd = np.array([len(k_near[i])/(k_reachDist[i].mean())
                       for i in range(self.size)])

        return lrd

    def _LOF_score(self, lrd, k_near):
        k_lrds = [lrd[k_near[idx]] for idx in range(self.size)]
        lof = np.array([k_lrds[idx].sum()/lrd[idx]
                       for idx in range(self.size)])

        return lof / self.k

    def predict(self, X):
        self.size = X.shape[0]
        dists, k_near, k_dist = self._KNN(X)
        reach_dist = self._reach_distance(dists, k_dist)
        lrd = self._LRD(reach_dist, k_near)
        output = self._LOF_score(lrd, k_near)

        return output


if __name__ == "__main__":
    orig_train_data = torchvision.datasets.MNIST("MNIST/", train=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]), target_transform=None, download=True)  # 下載並匯入MNIST訓練資料
    orig_test_data = torchvision.datasets.MNIST("MNIST/", train=False, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]), target_transform=None, download=True)  # 下載並匯入MNIST測試資料

    orig_train_label = orig_train_data.targets.numpy()
    orig_train_data = orig_train_data.data.numpy()
    orig_train_data = orig_train_data.reshape(60000, 28*28)

    orig_test_label = orig_test_data.targets.numpy()
    orig_test_data = orig_test_data.data.numpy()
    orig_test_data = orig_test_data.reshape(10000, 28*28)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=30)
    pca_data = pca.fit_transform(
        np.concatenate([orig_train_data, orig_test_data]))
    orig_train_data = pca_data[:len(orig_train_label)]
    orig_test_data = pca_data[len(orig_train_label):]

    orig_train_data, orig_train_label = resample_total(
        orig_train_data, orig_train_label, ratio=0.1)

    '''
    Initialize models
    '''
    knn_1 = KNN_AD(k=1)
    knn_5 = KNN_AD(k=5)
    knn_10 = KNN_AD(k=10)
    knns = [knn_1, knn_5, knn_10]

    knn_outputs = []
    knn_acc = []

    km_1 = KMeans_AD(n_clusters=1)
    km_5 = KMeans_AD(n_clusters=5)
    km_10 = KMeans_AD(n_clusters=10)
    kms = [km_1, km_5, km_10]

    km_outputs = []
    km_acc = []

    dist_ad = Distance_AD()
    metrics = ["CosineDistance", "L1Distance", "L2Distance",
               "ChebyshevDistance", "MahalanobisDistance"]

    dist_outputs = []
    dist_acc = []

    Lof = LOF()

    Lof_outputs = []
    Lof_acc = []

    for i in tqdm.tqdm(range(10)):
        train_data = orig_train_data[orig_train_label == i]
        test_data, test_label = resample(
            orig_test_data, orig_test_label, target_label=i, outlier_ratio=0.1)
        # [TODO] prepare training/testing data with label==i labeled as 0, and others labeled as 1
        test_label = np.where(test_label == i, 0, 1)
        # [TODO] implement methods
        for i in range(3):
            knns[i].fit(train_data)
            knn_outputs.append(knns[i].prediction(test_data))
            knn_acc.append(roc_auc_score(test_label, knn_outputs[-1]))

            kms[i].fit(train_data)
            km_outputs.append(kms[i].prediction(test_data))
            km_acc.append(roc_auc_score(test_label, km_outputs[-1]))

        for i in range(5):
            dist_outputs.append(dist_ad.predict(
                X=test_data, ref=train_data, metric=metrics[i]))
            dist_acc.append(roc_auc_score(test_label, dist_outputs[-1]))

        Lof_outputs.append(Lof.predict(test_data))
        Lof_acc.append(roc_auc_score(test_label, Lof_outputs[-1]))
        # [TODO] record ROC-AUC for each method
    # [TODO] print the average ROC-AUC for each method
    knn_acc = np.array(knn_acc)
    km_acc = np.array(km_acc)
    dist_acc = np.array(dist_acc)
    Lof_acc = np.array(Lof_acc)
    print(
        f"Average ROC-AUC of KNN:           (k=1)    {knn_acc[0:30:3].mean():.4f}, (k=5) {knn_acc[1:30:3].mean():.4f}, (k=10) {knn_acc[2:30:3].mean():.4f}")
    print(
        f"Average ROC-AUC of K-means:       (k=1)    {km_acc[0:30:3].mean():.4f}, (k=5) {km_acc[1:30:3].mean():.4f}, (k=10) {km_acc[2:30:3].mean():.4f}")
    print(
        f"Average ROC-AUC of Distance-Base: (Cosine) {dist_acc[0:50:5].mean():.4f}, (r=1) {dist_acc[1:50:5].mean():.4f}, (r=2)  {dist_acc[2:50:5].mean():.4f}, (r=inf) {dist_acc[3:50:5].mean():.4f}, (mahalanobis) {dist_acc[4:50:5].mean():.4f}")
    print(f"Average ROC-AUC of LOF:  {Lof_acc.mean():.4f}")

    # The TSNE.png of LOF
    test_data, test_label = resample(
        orig_test_data, orig_test_label, target_label=0, outlier_ratio=0.1)
    test_label = np.where(test_label == 0, 0, 1)
    Lof_score = Lof.predict(test_data)

    t_sne = TSNE()
    t_sne_out = t_sne.fit_transform(test_data)
    normal_idx = test_label == 0
    anomaly_idx = test_label == 1

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    im = axs[0].scatter(t_sne_out[:, 0], t_sne_out[:, 1], c=Lof_score)
    axs[0].set_title("predicted LOF score for normal digit=0", fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=20)

    axs[1].scatter(t_sne_out[normal_idx, 0],
                   t_sne_out[normal_idx, 1],
                   c='royalblue',
                   label='Normal')
    axs[1].scatter(t_sne_out[anomaly_idx, 0],
                   t_sne_out[anomaly_idx, 1],
                   c='orange',
                   label='Anomaly')
    axs[1].legend()
    axs[1].set_title("ground truth label for normal digit=0", fontsize=20)
    axs[1].tick_params(axis='both', which='major', labelsize=20)

    plt.colorbar(im, ax=axs[0])
    plt.savefig("TSNE.png", format="png")
    plt.show()

# %%
