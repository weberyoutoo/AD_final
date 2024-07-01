# %%
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from typing import Any

import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)


def data_plot(data, label, title):
    normal_data = data[label == 0]
    anomaly_data = data[label == 1]

    normal_idx = np.random.choice(normal_data.shape[0], size=10, replace=False)

    # Selecting "ALL" anomaly data, if the number of anomaly data <= 10
    if anomaly_data.shape[0] > 10:
        anomaly_idx = np.random.choice(
            anomaly_data.shape[0], size=10, replace=False)
    else:
        anomaly_idx = np.arange(anomaly_data.shape[0])

    fig, axs = plt.subplots(2, 1, figsize=(8, 16))
    x = np.arange(0, anomaly_data.shape[1])
    y = anomaly_data[anomaly_idx]
    for i in range(anomaly_idx.shape[0]):
        axs[0].plot(x, y[i], 'r')
    axs[0].set_title("Anomaly Sample")

    y = normal_data[normal_idx]
    for i in range(10):
        axs[1].plot(x, y[i], 'b')
    axs[1].set_title("Normal Sample")
    fig.suptitle(title)
    plt.show()


def KNN_AD(train_X, train_y, test_X, test_y, k=5):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(train_X, train_y)
    dist, _ = KNN.kneighbors(test_X)
    avg_dist = np.mean(dist, axis=1)

    return roc_auc_score(y_true=test_y, y_score=avg_dist)


def PCA_AD(train_X, test_X, test_y, k, return_recons=False):
    pca = PCA(n_components=k)
    pca.fit(train_X)
    test_recons = pca.inverse_transform(pca.transform(test_X))
    score = np.linalg.norm(test_X-test_recons, axis=1)
    if return_recons:
        return test_recons, roc_auc_score(y_true=test_y, y_score=score)

    return roc_auc_score(y_true=test_y, y_score=score)


def DFT(X, M=20, use_first_m=False, return_recons=False):
    ceo = np.fft.fft(X)
    mag_idx = np.zeros(X.shape[-1], dtype=bool)

    if use_first_m:       # use first M coes as features
        mag_idx[:M] = True
    else:                 # use first M/2 and last M/2 coes as featrues
        mag_idx[:int(np.ceil(M/2))] = True
        mag_idx[-int(np.floor(M/2)):] = True

    magnitudes = np.abs(ceo[:, mag_idx])
    ceo_selected = np.zeros_like(ceo)
    ceo_selected[:, mag_idx] = ceo[:, mag_idx]
    recons = np.fft.ifft(ceo_selected)

    if return_recons:
        return magnitudes, recons
    else:
        return magnitudes


def DWT(X, S, return_coes=False):
    S = 2**S
    levels = int(np.ceil(np.ma.log2(X.shape[-1])))
    size = 2**(levels) - 1
    X_padded = np.pad(X, pad_width=(
        (0, 0), (0, 2**levels-X.shape[-1])), mode='constant', constant_values=0)
    A_coes = np.pad(X_padded, ((0, 0), (0, size)),
                    mode='constant', constant_values=0)
    D_coes = np.zeros((X.shape[0], size))

    prev_begin = 0
    begin = X_padded.shape[-1]
    D_begin = 0

    for i in range(1, levels+1):
        len = levels - i

        A_coes[:, begin:begin+2 **
               len] = (A_coes[:, prev_begin+1:begin:2] + A_coes[:, prev_begin:begin:2]) / 2
        D_coes[:, D_begin:D_begin+2 **
               len] = (A_coes[:, prev_begin+1:begin:2] - A_coes[:, prev_begin:begin:2]) / 2

        prev_begin = begin
        begin += 2**len
        D_begin += 2**len

    out_A_coes = np.expand_dims(A_coes[:, -1], axis=1)
    out_D_coes = D_coes[:, -1:-S:-1]
    if return_coes:
        return np.concatenate([out_A_coes, out_D_coes], axis=1),  A_coes[:, -size:], D_coes

    return np.concatenate([out_A_coes, out_D_coes], axis=1)


class AD:
    def __init__(self, func=None) -> None:
        self.process_func = func

    def __call__(self, train_X, train_y, test_X, test_y, k=5, arg=None) -> float:
        if self.process_func == None:
            train_proc = train_X
            test_proc = test_X
        else:
            train_proc = self.process_func(train_X, arg)
            test_proc = self.process_func(test_X, arg)

        return KNN_AD(train_proc, train_y, test_proc, test_y, k=k)


def resample(data, label, outlier_ratio=0.01, target_label=0):
    """
    Resample the data to balance classes.

    Parameters:
        data: np.array, shape=(n_samples, n_features)
            Input data.
        label: np.array, shape=(n_samples,)
            Labels corresponding to the data samples.
        outlier_ratio: float, optional (default=0.01)
            Ratio of outliers to include in the resampled data.
        target_label: int, optional (default=0)
            The label to be treated as normal.

    Returns:
        new_data: np.array
            Resampled data.
        new_label: np.array
            Resampled labels.
    """
    new_data = []
    new_label = []
    for i in [1, -1]:
        if i != target_label:
            i_data = data[label == i]
            target_size = len(data[label == target_label])
            num = target_size * outlier_ratio
            idx = np.random.choice(
                list(range(len(i_data))), int(num), replace=False
            )
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx)) * 1)
        else:
            new_data.append(data[label == i])
            new_label.append(np.ones(len(data[label == i])) * 0)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label


if __name__ == '__main__':
    Raw_AD = AD()
    DFT_AD = AD(DFT)
    DWT_AD = AD(DWT)

    categorys = ["Wafer", "ECG200"]
    for category in categorys:
        # Load the data
        # category = "ECG200" # Wafer / ECG200
        print(f"Dataset: {category}")
        train_data = pd.read_csv(
            f'./{category}/{category}_TRAIN.tsv', sep='\t', header=None).to_numpy()
        test_data = pd.read_csv(
            f'./{category}/{category}_TEST.tsv', sep='\t', header=None).to_numpy()

        train_label = train_data[:, 0].flatten()
        train_data = train_data[:, 1:]
        train_data, train_label = resample(
            train_data, train_label, outlier_ratio=0.0, target_label=1)

        test_label = test_data[:, 0].flatten()
        test_data = test_data[:, 1:]
        test_data, test_label = resample(
            test_data, test_label, outlier_ratio=0.1, target_label=1)

        # 1
        data_plot(test_data, test_label, category+" Raw")

        # 2
        print(
            f"ROC-AUC of raw data: {Raw_AD(train_data, train_label, test_data, test_label):.4f}\n")

        # 3
        pca_auc = []
        max_K = int(np.ceil(test_data.shape[-1]/100))*25

        for i in range(1, max_K+1):
            pca_auc.append(PCA_AD(train_data, test_data, test_label, k=i))
            print(f"ROC-AUC of PCA (k = {i}) : {pca_auc[-1]:.4f}")

        best_pca = np.argmax(pca_auc)
        print(f"\nBest PCA k={best_pca+1}, ROC-AUC : {pca_auc[best_pca]:.4f}")

        test_recons, _ = PCA_AD(
            train_data, test_data, test_label, k=best_pca, return_recons=True)
        data_plot(test_recons, test_label,
                  category+f" PCA, k={best_pca+1}")

        # 4
        dft_aucs = []
        max_M = int(np.ceil(test_data.shape[-1]/100))*50

        for i in range(1, max_M+1):
            dft_aucs.append(DFT_AD(train_data, train_label,
                            test_data, test_label, arg=i))
            print(f"ROC-AUC of DFT (M = {i}): {dft_aucs[-1]:.4f}")

        best_dft = np.argmax(dft_aucs)
        print(
            f"\nBest DFT M={best_dft+1}, ROC-AUC : {dft_aucs[best_dft+1]:.4f}")

        test_mag, test_recons = DFT(
            test_data, M=best_dft+1, return_recons=True)
        data_plot(test_recons, test_label,
                  category+f" DFT, M={best_dft+1}")

        # 5
        dwt_aucs = []
        max_level = int(np.ceil(np.ma.log2(test_data.shape[-1]))+1)

        for i in range(1, max_level+1):
            dwt_aucs.append(DWT_AD(train_data, train_label,
                            test_data, test_label, arg=i))
            print(f"ROC-AUC of DWT (S = {2**i}): {dwt_aucs[-1]:.4f}")

        best_dwt = np.argmax(dwt_aucs)
        print(
            f"\nBest DWT S={2**(best_dwt+1)}, ROC-AUC : {dwt_aucs[best_dwt]:.4f}")

        # Bonus
        print("\nBonus=========================================================================================================")
        raw_aucs = []
        dft_aucs = []
        dwt_aucs = []

        for k in range(1, 11):
            raw_aucs.append(Raw_AD(train_data, train_label,
                            test_data, test_label, k=k))
            print(f"ROC-AUC of raw data (k = {k}): {raw_aucs[-1]:.4f}")
        print(
            f"Best raw k={np.argmax(raw_aucs)+1}, ROC-AUC : {np.max(raw_aucs):.4f}\n")

        max_M = int(np.ceil(test_data.shape[-1]/100))*50
        for k in range(1, 11):
            for i in range(1, max_M+1):
                dft_aucs.append(DFT_AD(train_data, train_label,
                                test_data, test_label, k=k, arg=i))
                print(
                    f"ROC-AUC of DFT data (k = {k}, M = {i}): {dft_aucs[-1]:.4f}")
        dft_aucs = np.array(dft_aucs).reshape(10, -1)
        best_dft = np.unravel_index(
            np.argmax(dft_aucs, axis=None), dft_aucs.shape)
        print(
            f"Best DFT k={best_dft[0]+1}, M={best_dft[1]+1}, ROC-AUC : {np.max(dft_aucs):.4f}\n")

        max_level = int(np.ceil(np.ma.log2(test_data.shape[-1]))+1)
        for k in range(1, 11):
            for i in range(1, max_level+1):
                dwt_aucs.append(DWT_AD(train_data, train_label,
                                test_data, test_label, k=k, arg=i))
                print(
                    f"ROC-AUC of DWT data (k = {k}, S = {2**i}): {dwt_aucs[-1]:.4f}")
        dwt_aucs = np.array(dwt_aucs).reshape(10, -1)
        best_dwt = np.unravel_index(
            np.argmax(dwt_aucs, axis=None), dwt_aucs.shape)
        print(
            f"Best DWT k={best_dwt[0]+1}, S={2**(best_dwt[1]+1)}, ROC-AUC : {np.max(dwt_aucs):.4f}\n")
