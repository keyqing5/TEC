import os
import pdb
import sys
sys.path.append("..")
import numpy as np
from teco.te_hic import Partition_One_Layer

def get_txt_file_paths(directory):
    path_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            full_path = os.path.join(directory, filename)
            path_list.append(full_path)
    return path_list

def main_proc_single(input_file, objective="KL", output_file="", eta=3.0, eta2=1,dbl=False):
    mod = Partition_One_Layer(objective=objective, sparsification="affinity",
                              affinity="precomputed", kernel_gamma=20, n_neighbors=20,
                              eta_mode="coefficient", eta1=1, eta=eta,
                              eta2=eta2, generate_tsv=True, save_tsv=output_file)
    matrix = np.loadtxt(input_file)
    # mod.fit(matrix)
    mod.fit_v2(matrix, double_layer=dbl)

def main_proc(input_directory, output_root, objective="KL", eta=3.0, eta2=1,dbl=False):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # get file names
    # for each file
    file_name = os.path.basename(input_directory)
    output_dir = os.path.join(output_root, f"{file_name[:-4]}_eta_{eta}_eta2_{eta2}")
    main_proc_single(input_file=input_directory, output_file=output_dir, objective=objective, eta=eta, eta2=eta2,dbl=dbl)

if len(sys.argv) < 5:
    print("Usage: python test_our.py <input directory> <output_root_dir> <objective> <eta> <eta2> <double_layer>")
    sys.exit(1)

eta2_v = 1
dbl = False

if len(sys.argv) >= 6:
    eta2_v = float(sys.argv[5])

if len(sys.argv)>=7:
    tmp = sys.argv[6]
    if tmp=='0' or tmp.strip().lower() == 'false':
        dbl = False
    else:
        dbl = True

main_proc(input_directory=sys.argv[1], output_root=sys.argv[2], objective=sys.argv[3], eta=float(sys.argv[4]), eta2=eta2_v, dbl=dbl)

print("Over..")
