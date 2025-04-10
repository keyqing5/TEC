import pdb

import numpy as np
import subprocess
from pandas.core.frame import DataFrame
import os
import sys
sys.path.append("..")
from teco.te_hic import Partition_One_Layer
import re


def convert_domains2bound(domains):
    boundaries = [domains.start, domains.end]
    for leaf in domains.leaves:
        left = leaf.start
        right = leaf.end
        boundaries.append(left)
        boundaries.append(right)
    # remove redundant elements
    boundaries = list(set(boundaries))
    # remove adjacent elements
    boundaries.sort()
    boundaries_copy = [boundaries[0]]
    for i in range(len(boundaries)-1):
        if boundaries[i] + 1 == boundaries[i+1]:
            continue
        else:
            boundaries_copy.append(boundaries[i+1])
    # get bound list
    # except the start and end bin, all the indices indicate left bounds
    boundaries = boundaries_copy
    if boundaries[-1] == domains.end:
        boundaries.remove(domains.end)
    right_bound = [i-1 for i in boundaries[1:]]
    right_bound.append(domains.end)
    bound = (boundaries,right_bound)
    return bound

def trans_2_tab_mode(input_file, output_file="output_file.txt"):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        new_line = line.replace(' ', '\t')
        new_lines.append(new_line)

    with open(output_file, 'w') as file:
        file.writelines(new_lines)

def read_file_dirs(file_path):
    path_list = []

    with open(file_path, 'r') as file:
        for line in file:
            path = line.strip()  # 去除行末尾的换行符和空格
            path_list.append(path)
    return path_list

def generate_tsv_file(bound, tsv_root_dir="", start_pos=1, data_file_name="sim"):
    file_name = tsv_root_dir +"/"+ data_file_name + ".tsv"
    # format: chr_name start_bin(index+1) (start_bin-1)*resolution start_bin*resolution chr_name end_bin (
    # index+1) (end_bin-1)*resolution end_bin*resolution form the content
    output = []
    data_stat = data_file_name
    if data_stat == "sim":
        chr_name = "sim"
        resolution_s = 40
    else:
        chr_start = data_file_name.find("chr")
        chr_pos = data_file_name.find("_KR")
        chr_name = data_file_name[chr_start:chr_pos]
        # resolution_pos = data_file_name.find("R")
        # resolution_pos_2 = data_file_name.find("k")
        # resolution_s = data_file_name[resolution_pos + 2:resolution_pos_2]
        match = re.search(r'(\d+)kb', data_file_name)
        resolution_s = match.group(1)
    # if resolution_s[-1] == 'k':
    #     resolution_s = resolution_s.strip("k")
    resolution = int(resolution_s) * 1000
    # start_pos = start_pos + bound[0][0]
    bound = ([int(x) for x in bound[0]], [int(x) for x in bound[1]])
    for b_i in range(len(bound[0])):
        line = [chr_name, (bound[0][b_i] + start_pos),
                (bound[0][b_i] + start_pos - 1) * resolution,
                (bound[0][b_i] + start_pos) * resolution,
                chr_name, (bound[1][b_i] + start_pos),
                (bound[1][b_i] + start_pos - 1) * resolution,
                (bound[1][b_i] + start_pos) * resolution]
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
    #delete temp file
    os.remove(tsv_root_dir + "temp.tsv")
    print("write boundaries successfully: " + file_name)

def generate_tsv_file2(domains, tsv_root_dir="", start_pos=1, data_file_name="sim"):
    file_name = tsv_root_dir + "/" + data_file_name + ".tsv"
    # format: chr_name start_bin(index+1) (start_bin-1)*resolution start_bin*resolution chr_name end_bin (
    # index+1) (end_bin-1)*resolution end_bin*resolution form the content
    output = []
    data_stat = data_file_name
    if data_stat == "sim":
        chr_name = "sim"
        resolution_s = 40
    else:
        chr_start = data_file_name.find("chr")
        chr_pos = data_file_name.find("_KR")
        chr_name = data_file_name[chr_start:chr_pos]
        # resolution_pos = data_file_name.find("R")
        # resolution_pos_2 = data_file_name.find("k")
        # resolution_s = data_file_name[resolution_pos + 2:resolution_pos_2]
        match = re.search(r'(\d+)kb', data_file_name)
        resolution_s = match.group(1)
    # if resolution_s[-1] == 'k':
    #     resolution_s = resolution_s.strip("k")
    resolution = int(resolution_s) * 1000
    lefts = []
    rights = []
    left = domains.start
    right = domains.end
    line = [chr_name, (left + start_pos),
            (left + start_pos - 1) * resolution,
            (left + start_pos) * resolution,
            chr_name, (right + start_pos),
            (right + start_pos - 1) * resolution,
            (right + start_pos) * resolution]
    output.append(line)
    for leaf in domains.leaves:
        left = leaf.start
        right = leaf.end
        line = [chr_name, (left + start_pos),
                (left + start_pos - 1) * resolution,
                (left + start_pos) * resolution,
                chr_name, (right + start_pos),
                (right + start_pos - 1) * resolution,
                (right + start_pos) * resolution]
        output.append(line)
        lefts.append(left)
        rights.append(right)
    # write file
    output = DataFrame(output)
    with open(tsv_root_dir + "temp.tsv", 'w') as write_tsv:
        write_tsv.write(output.to_csv(sep='\t', index=False))
    with open(tsv_root_dir + "temp.tsv", 'r') as f:
        with open(file_name, 'w') as f1:
            next(f)  # skip header line
            for line in f:
                f1.write(line)
    # delete temp file
    os.remove(tsv_root_dir + "temp.tsv")
    print("write boundaries successfully: " + file_name)

def trans_file(filename, start_pos):
    """
    trans .TAD file to .domains file
    """
    # read bound
    bound = ([],[])
    with open(filename, 'r') as file:
        for line in file:
            numbers = line.split()  # 按空格分割
            if numbers:
                bound[0].append(int(numbers[0])-1)
                bound[1].append(int(numbers[-1])-1)
    filename = filename.replace('.TAD', '_dedoc')
    generate_tsv_file(bound=bound, tsv_root_dir=os.path.dirname(filename), start_pos=start_pos, data_file_name=os.path.basename(filename))
    return filename


def resplit_filename(output_dir):
    root_dir = os.path.dirname(output_dir)
    # datafile_name = os.path.basename(output_dir)
    return root_dir

def run_our_mod(matrix, datafile_name,objective="KL", output_file="",eta=3.0, eta2=1, start_pos=1,dbl=False):
    root_dir = resplit_filename(output_dir=output_file)
    mod = Partition_One_Layer(objective=objective, sparsification="affinity",
                              affinity="precomputed", kernel_gamma=20, n_neighbors=20,
                              eta_mode="coefficient", eta1=1, eta=eta, start_pos=start_pos,
                              eta2=eta2, generate_tsv=True, save_tsv=root_dir,data_file_name=datafile_name+"_"+objective+"_eta="+str(eta)+"_eta2="+str(eta2),data_stat="real")
    mod.fit_v2(matrix, double_layer=dbl)

def main_run(mode,input_file, output_file,start_pos=1,eta=3.0, eta2=1, resolution=40, objective="KL", dedoc_dir="/home/grads/qiusliang2/deDoc2/",dbl=True):
    matrix = np.loadtxt(input_file)
    datafile_name = os.path.basename(input_file)
    datafile_name = datafile_name[:-4]
    if mode =="our":
        run_our_mod(matrix=matrix, objective=objective, output_file=output_file,eta=eta, eta2=eta2, start_pos=start_pos,dbl=dbl, datafile_name=datafile_name)
    else:
        print("Wrong mode word, please retry!")

if len(sys.argv) < 6:
    print("Usage: python test_real_hic.py <mode> <input> <output_root_dir> <start pos> <resolution> <objective> <eta> <eta2> <double_layer>")
    sys.exit(1)


objective="KL"
eta = 3
eta2 = 1
dbl = False

if len(sys.argv) >=7:
    objective = sys.argv[6]
if len(sys.argv) >=8:
    eta = float(sys.argv[7])
if len(sys.argv) >=9:
    eta2 = float(sys.argv[8])
if len(sys.argv)>=10:
    tmp = sys.argv[9]
    if tmp=='0' or tmp.strip().lower() == 'false':
        dbl = False
    else:
        dbl = True

main_run(mode=sys.argv[1], input_file=sys.argv[2], output_file=sys.argv[3],start_pos=int(sys.argv[4]), resolution=int(sys.argv[5]),eta=eta, eta2=eta2, objective=objective,dbl=dbl)