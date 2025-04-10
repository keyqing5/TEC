"""
2024/07/04
"""
import numpy as np
from pandas.core.frame import DataFrame
import sys
sys.path.append("..")
from tecu import graph_metric
from tecu.TECU import SEAT    # commented for local debug
from tecu.TECU import Node
import pdb
import re

import sys
import os


class Partition_One_Layer():

    def __init__(self, objective='SE', sparsification="knn_neighbors", affinity="precomputed", n_neighbors=30,kernel_gamma=20,eta_mode="coefficient",eta=3.0,eta2=0.5,k_scal=1, eta1=1, generate_tsv=False, save_tsv="",data_file_name="sim", data_stat="sim", start_pos=1):
        self.objective = objective
        self.eta_mode = eta_mode
        self.eta = eta
        self.eta2 = eta2
        self.partition_node_list = []
        self.affinity = affinity
        self.gamma = kernel_gamma
        self.sparsification = sparsification
        self.k_scal = k_scal
        self.eta1 = eta1
        self.n_neighbors = n_neighbors
        self.objective_paras = None
        self.boundaries = None
        self.generate_tsv = generate_tsv
        self.save_tsv = save_tsv
        self.data_file_name = data_file_name
        self.data_stat = data_stat
        self.start_pos = start_pos

        self.second_boundaries = ([], [])

    def __prepare__(self, input_matrix):
        self.mod = SEAT(affinity=self.affinity,
                     kernel_gamma=self.gamma,
                     sparsification=self.sparsification, n_neighbors=self.n_neighbors,
                     objective=self.objective,
                     strategy="bottom_up", eta=self.eta, eta2=self.eta2,
                     eta_mode=self.eta_mode,__verbose__=True, plot_cluster_map=True)
        self.objective_paras = self.mod.objective_paras
        self.mod.fit_v3(input_matrix)
        self.matrix = self.mod.aff_m

    def _fill_dp_table(self):
        # M, m, d, log_d, d_log_d, vG, log_vG, sparse_m, d_square = self.mod.se_tree.knn_graph_stats
        M, m, d, log_d, d_log_d, vG, log_vG, sparse_m, d_square, _ = self.mod.se_tree.knn_graph_stats
        aff_M, aff_m, aff_d, aff_log_d, aff_d_log_d, aff_vG, aff_log_vG, aff_sparse_m, aff_d_square, _ = self.mod.se_tree.aff_graph_stats
        root = Node(graph_stats=self.mod.se_tree.knn_graph_stats, node_id=self.mod.se_tree.update_node_id(), children=[],
                         vs=list(range(self._N_)),objective_paras=self.objective_paras, is_leaf=False, __verbose__=True
                         )
        for i in range(self._N_):  # k=0
            # only one leaf
            node = Node(graph_stats=self.mod.se_tree.knn_graph_stats,
                             node_id=self.mod.se_tree.update_node_id(), children=[], vs=list(range(i+1)),
                             parent=root, objective_paras=self.objective_paras)
            if self.objective == "SE":
                node_V = graph_metric.get_v(aff_M, sparse_m, node.vs)
                vertices_se = graph_metric.get_node_vertices_se(vG, np.sum(aff_d_log_d[node.vs]), node_V)
                node_se = node.se + vertices_se
            else:
                node_se = - node.se
            self.dp_table[i, 0] = node_se
        for k in range(1, self._K_):
            self.dp_table[:k, k] = float("inf")
            for i_r in range(k, self._N_):
                se_temp = np.full((self._N_), float("inf"))
                for i in range(i_r):
                    node = Node(graph_stats=self.mod.se_tree.knn_graph_stats,
                                     node_id=self.mod.se_tree.update_node_id(), children=[],
                                     vs=list(range(i + 1, i_r+1)),
                                     parent=root, objective_paras=self.objective_paras)
                    if self.objective == "SE":
                        node_V = graph_metric.get_v(aff_M, sparse_m, node.vs)
                        vertices_se = graph_metric.get_node_vertices_se(vG, np.sum(
                            aff_d_log_d[node.vs]), node_V)
                        node_se = node.se + vertices_se
                    else:
                        node_se = - node.se
                    se_temp[i] = self.dp_table[i, k-1] + node_se
                self.dp_table[i_r, k] = min(se_temp)
                self.dp_table_i[i_r, k] = np.argmin(se_temp)

    def _backtrace(self):
        min_entro = min(self.dp_table[self._N_-1, 1:])
        print("minimum entropy: ", min_entro)
        k_op = np.argmin(self.dp_table[self._N_-1, 1:])+1
        if k_op == 0:
            print("bound: [0, N-1]")
            return [0, self._N_-1], min_entro
        i_pre = self.dp_table_i[self._N_ - 1, k_op]
        i_list = [self._N_-1, i_pre]
        for k in range(k_op-1, 0, -1):
            i_pre = self.dp_table_i[i_pre, k]
            if i_pre < 0:
                pdb.set_trace()
            i_list.append(i_pre)
        i_list.append(0)
        i_list.reverse()
        print("bound: ", i_list)
        return i_list, min_entro

    def _generate_labels(self):
        labels = np.full(self._N_, -1)
        left_bound = 0
        for k in range(len(self.bound)-1):
            right_bound = self.bound[k+1]
            labels[left_bound:right_bound+1] = k
            left_bound = right_bound + 1
        return labels

    def _generate_left_right_bound(self):
        left_bound = 0
        bound_list = ([left_bound],[])
        for i in range(1,len(self.bound)-1):
            right_bound = self.bound[i]
            bound_list[1].append(right_bound)
            bound_list[0].append(right_bound+1)
        bound_list[1].append(self._N_-1)
        return bound_list


    def fit(self, input_matrix):
        self.__prepare__(input_matrix)
        self._N_ = self.matrix.shape[0]
        self._K_ = int(self._N_*self.k_scal)
        self.dp_table = np.full((self._N_, self._K_), float("inf"))
        self.dp_table_i = np.full((self._N_, self._K_), -1)  # save the right bounds
        self._fill_dp_table()
        self.bound, min_entro = self._backtrace()
        # order = self._order()
        # self._get_clubs(order)
        self.labels_ = self._generate_labels()
        # self.se_score = self._get_entropy()
        self.boundaries = self.process_bound()
        if self.generate_tsv:
            self.generate_tsv_file()

    def fit_sub(self, input_matrix):
        self.__prepare__(input_matrix)
        self._N_ = self.matrix.shape[0]
        self._K_ = int(self._N_*self.k_scal)
        self.dp_table = np.full((self._N_, self._K_), float("inf"))
        self.dp_table_i = np.full((self._N_, self._K_), -1)  # save the right bounds
        self._fill_dp_table()
        self.bound, min_entro = self._backtrace()
        # order = self._order()
        # self._get_clubs(order)
        self.labels_ = self._generate_labels()
        # self.se_score = self._get_entropy()
        bound = self.process_bound()
        return bound

    def fit_v2(self, matrix, double_layer=False):
        # 执行第一层划分
        bound = self.fit_sub(matrix)

        if double_layer:
            # 遍历每个父区间
            for i in range(len(bound[0])):
                left = bound[0][i]
                right = bound[1][i]

                if right - left<=3:
                    self.second_boundaries[0].append(left)
                    self.second_boundaries[1].append(right)
                    continue
                # 提取子矩阵
                sub_matrix = matrix[left:right + 1, left:right + 1]
                # 计算该子矩阵的boundaries
                sub_boundaries = self.fit_sub(sub_matrix)
                # 将父区间的左、右边界和内部boundaries保存到double_boundaries
                self.second_boundaries[0].extend([x+left for x in sub_boundaries[0]])
                self.second_boundaries[1].extend([x+left for x in sub_boundaries[1]])
            bound[0].extend(self.second_boundaries[0])
            bound[1].extend(self.second_boundaries[1])
        self.boundaries = bound
        if self.generate_tsv:
            self.generate_tsv_file()
        return

    # def _get_entropy(self):
    #     entro_sum = 0
    #     for k in range(len(self.partition_node_list)):
    #         entro_sum = entro_sum + self.partition_node_list[k].se
    #     return entro_sum

    # def plot_heatmap_fig(self,output_title):
    #     bound = self._generate_left_right_bound()
    #     plot_heatmap.heatmap(self.matrix, boundary=bound, max=1, all=False, out_path=output_title)

    def generate_true(self, filename="simulate_structure.txt"):
        f = open(filename, "r")
        flag = True
        input_struc = []
        while flag is True:
            content = f.readline()
            if content == '':
                flag = False
                break
            content = content.strip(' \n')
            list_i = content.split(' ')
            arr_tmp = np.array(list_i).astype(dtype=int)
            arr_tmp = arr_tmp - 1
            # arr_tmp.tolist()
            input_struc.append(arr_tmp)
        f.close()
        # input_struc = input_struc - 1
        n_samples = input_struc[-1][-1] + 1
        label_true = np.zeros(shape=(n_samples,))
        for i in range(len(input_struc)):
            start = input_struc[i][0]
            end = input_struc[i][-1]
            label_true[start:end + 1, ] = i
        return label_true

    def process_bound(self):
        labels = self.labels_
        left_bound = 0
        bound = ([left_bound], [])
        for i in range(len(labels) - 1):
            flag = labels[i]
            if labels[i + 1] != flag:
                right_bound = i
                bound[1].append(right_bound)
                left_bound = right_bound + 1
                bound[0].append(left_bound)
        bound[1].append(len(labels) - 1)
        return bound

    def generate_tsv_file(self):
        bound = self.boundaries
        tsv_root_dir = self.save_tsv
        if not os.path.exists(tsv_root_dir):
            os.makedirs(tsv_root_dir)
        file_name = tsv_root_dir +"/"+ self.data_file_name + ".tsv"
        # format: chr_name start_bin(index+1) (start_bin-1)*resolution start_bin*resolution chr_name end_bin (
        # index+1) (end_bin-1)*resolution end_bin*resolution form the content
        output = []
        if self.data_stat == "sim":
            chr_name = "sim"
            resolution_s = 40
        else:
            chr_start = self.data_file_name.find("chr")
            chr_pos = self.data_file_name.find("_", chr_start)
            chr_name = self.data_file_name[chr_start:chr_pos]
            match = re.search(r'(\d+)kb', self.data_file_name)
            resolution_s = match.group(1)
            # if resolution_s[-1] == 'k':
            #     resolution_s = resolution_s.strip("k")
        # resolution = int(resolution_s) * 1000
        # if resolution_s[-1] == 'k':
        #     resolution_s = resolution_s.strip("k")

        resolution = int(resolution_s) * 1000

        # start_pos = start_pos + bound[0][0]
        for b_i in range(len(bound[0])):
            line = [chr_name, (bound[0][b_i] + self.start_pos),
                    (bound[0][b_i] + self.start_pos - 1) * resolution,
                    (bound[0][b_i] + self.start_pos) * resolution,
                    chr_name, (bound[1][b_i] + self.start_pos),
                    (bound[1][b_i] + self.start_pos - 1) * resolution,
                    (bound[1][b_i] + self.start_pos) * resolution]
            output.append(line)
        # write file
        output = DataFrame(output)
        with open(tsv_root_dir + "temp.tsv", 'w') as write_tsv:
            write_tsv.write(output.to_csv(sep='\t', index=False))
        with open(tsv_root_dir + "temp.tsv", 'r') as f:
            with open(file_name, 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)
        print("write boundaries successfully: " + file_name)
