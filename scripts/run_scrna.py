"""
1. read dataset list
2. load data, meta file
3. for each dataset, run
4. save
"""
import pdb
import numpy as np
import random
import math
import umap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score, f1_score, \
    pairwise_distances
import sys
sys.path.append("..")
from tecu.TECU import SEAT as SEAT2
import pandas as pd
import os
import argparse
from scipy.stats import pearsonr


def convert_labels(labels):
    labels_name = labels.unique()
    K_true = labels_name.shape[0]
    labels_tmp = labels.values
    num = 0
    for label_name in labels_name:
        pos = np.where(labels_tmp == label_name)
        labels_tmp[pos] = num
        num += 1
    return labels_tmp, K_true


def get_metric_vals(df, key_word="optimal_cluster"):
    predicted_labels = df[key_word]
    true_labels = df['ground_truth']

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    return [ari, nmi, v_measure]


def compute_pearson_similarity_matrix(X):
    n_cells, n_genes = X.shape
    similarity_matrix = np.zeros((n_cells, n_cells))

    for i in range(n_cells):
        for j in range(i, n_cells):
            corr, _ = pearsonr(X[i], X[j])
            similarity_matrix[i, j] = corr
            similarity_matrix[j, i] = corr

    return similarity_matrix


def make_compar(X, n_clusters, result_dir_root, data_name,  X_umap=None, eta=3.0, eta2=1.0,
                sparse="knn_neighbors_from_X", affinity="gaussian_kernel", merge_flag=True):
    n_neighbors = int(X.shape[0] / n_clusters)
    if X_umap is None:
        X_umap = X

    plots = []

    # SEAT2: KL variants
    bottom_up2 = SEAT2(affinity=affinity, kernel_gamma=None, sparsification=sparse,
                       n_neighbors=n_neighbors, objective="KL", strategy="bottom_up", eta=eta, eta1=1,
                       eta_mode="coefficient", eta2=eta2, merge_layers=merge_flag, plot_cluster_map=True,
                       __verbose__=False, max_k=n_clusters)
    bottom_up2.fit_v3(X_umap)
    method_prefix = f'SEAT2_knn(TE_{eta}_{eta2})'
    y_seat2 = bottom_up2.labels_
    y_seat2_clubs = bottom_up2.clubs
    y_seat2_k = bottom_up2.ks_clusters['K=' + str(n_clusters)]
    if len(set(y_seat2_k.tolist())) == 1:
        y_seat2_k = y_seat2_clubs
    plots.extend([
        (y_seat2, f'{method_prefix} labels'),
        (y_seat2_clubs, f'{method_prefix} clubs'),
        (y_seat2_k, f'{method_prefix} predefine K')
    ])


    results = [item[0] for item in plots]
    titles = [item[1] for item in plots]
    df = pd.DataFrame({title: pd.Series(result) for title, result in zip(titles, results)})

    df.to_csv(os.path.join(result_dir_root, f"{data_name}_cluster_result.csv"), index=False)



def run(data_name,data_dir,result_dir_root, eta=3.0, eta2=1.0,
        sparse="knn_neighbors_from_X", affinity="gaussian_kernel",n_clusters=30,umap=False,merge_flag=True):
    X = np.loadtxt(data_dir,dtype=float,delimiter=',')
    if umap:
        X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    else:
        X_umap = X

    # result_dir_root_t = os.path.join(result_dir_root, f"{sparse}_{aff}")
    # os.makedirs(result_dir_root_t, exist_ok=True)
    result_dir_root_t = result_dir_root
    os.makedirs(result_dir_root_t, exist_ok=True)


    make_compar(
        X=X, n_clusters=n_clusters,
        result_dir_root=result_dir_root_t,
        data_name=data_name,
        X_umap=X_umap,
        eta=eta,
        eta2=eta2,
        sparse=sparse,
        affinity=affinity,
        merge_flag=merge_flag
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering algorithms with specified data name.")

    # 添加 data_name 参数
    parser.add_argument('--data_name', type=str, required=True, help="The name of the dataset.")
    parser.add_argument('--data_dir', type=str, required=True, help="The directory of the dataset.")
    parser.add_argument('--result_dir', type=str, required=True, help="The directory of the output.")
    parser.add_argument('--eta', type=float,default=3.0, required=False, help="The value of alpha1.")
    parser.add_argument('--eta2', type=float, default=0.85,required=False, help="The value of alpha2.")
    parser.add_argument('--n_clusters', type=int, default=30, required=False, help="Numbers of clusters.")
    parser.add_argument('--sparse', type=str, default="knn_neighbors_from_X", required=False, help="Parameter of sparsification.")
    parser.add_argument('--affinity', type=str, default="gaussian_kernel", required=False, help="Parameter of constructing dense graph.")
    parser.add_argument('--umap', type=bool, default=False, required=False,
                        help="Whether to use UMAP to process input matrix.")
    parser.add_argument('--merge_layer', type=bool, default=True, required=False,
                        help="Whether to use merging_layer.")

    args = parser.parse_args()

    # 获取 data_name 参数的值
    data_name = args.data_name
    data_dir = args.data_dir
    result_dir = args.result_dir
    eta = float(args.eta)
    eta2 = float(args.eta2)
    n_clusters = int(args.n_clusters)
    sparse = args.sparse
    affinity = args.affinity
    umap = args.umap
    merge_flag = args.merge_layer



    run(data_name=data_name,
        data_dir=data_dir,
        result_dir_root=result_dir,
        eta=eta,
        eta2=eta2,
        n_clusters=n_clusters,
        sparse= sparse,
        affinity=affinity,
        umap=umap,
        merge_flag=merge_flag
    )
