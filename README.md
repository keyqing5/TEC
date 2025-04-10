# Topology Entropy: Enhancing Graph Partitioning for TAD Identification and Single-Cell Clustering

The topology entropy-based graph partitioning methods 
for TAD identification and single cell clustering. 
## Requirements
Please see requirements.txt

## Input and output

### TEC-O
The input should be symmetric matrix. Example matrices are in the folder "example_data/hic".
For generation of simulated Hi-C matrix, please visit https://github.com/keyqing5/InfoTAD.

An example output of the tsv file is shown below (resolution=1kb):
```
chr1	1	0       1000	chr1	44	43000	44000

chr1	9	8000	9000	chr1	16	15000	16000

chr1	17	16000	17000	chr1	44	43000	44000
```

### TEC-U
The input can be low-dimensional or symmetric matrix. If your input is a high dimensional matrix,
please reduce the dimension using UMAP or other tools.

The example data is the p3cl dataset mentioned in the paper. 
We provide the processed dataset as example. 
You can find it as example_data/scrna/p3cl_count_umap.csv

The output is dataframe containing the cluster id of each cell.

## Quick Start

### TEC-O


For simulated data, you can run command like:
```
cd tec/scripts
python test_our2.py <absolute input data directory> <output directory> <objective> <eta> <eta2> <double_layer>
```
The optional value for '<objective>' is 'KL' for topology entropy.
The values 'eta' and 'eta2' correspond to '\alpha_1' and '\alpha_2' in the paper.
We recommend the range of 'eta' as 2~5 and the range of 'eta2' as 0.8~1.1.
We use both parameters to control the size of partition.
The optionla values for 'double_layer' are 'True' or 'False', indicating whether
to generate 2-level structures.

An example of the command is:
```
cd tec/scripts
python test_our2.py ../example_data/hic/simulation_data.txt ../ KL 2 0.85
```

For real Hi-C contact matrix (an example is example_data/hic/chr19_25kb_99_200.txt),
you can run command like:
```
cd tec/scripts
python test_real_hic2.py <mode> <input> <output_root_dir> <start pos> <resolution> <objective> <eta> <eta2> <double_layer>
```

An example of the command is:
```
cd tec/scripts
python test_real_hic2.py our ../example_data/hic/chr19_25kb_99_200.txt ../ 100 25 KL 3 0.95
```



### TEC-U


An example of command is:
```
cd tec/scripts
python run_scrna.py --data_name p3cl --data_dir ../example_data/scrna/p3cl_count_umap_mat.csv --result_dir ../ --eta 3 --eta2 0.9
```

#### Parameters
'--data_name', type=str, required=True, help="The name of the dataset."

'--data_dir', type=str, required=True, help="The directory of the dataset."

'--result_dir', type=str, required=True, help="The directory of the output."

'--eta', type=float,default=3.0, required=False, help="The value of alpha1."

'--eta2', type=float, default=0.85,required=False, help="The value of alpha2."

'--n_clusters', type=int, default=30, required=False, help="Numbers of clusters."

'--sparse', type=str, default="knn_neighbors_from_X", required=False, help="Parameter of sparsification."

'--affinity', type=str, default="gaussian_kernel", required=False, help="Parameter of constructing dense graph."

'--umap', type=bool, default=False, required=False,
                        help="Whether to use UMAP to process input matrix."

'--merge_layer', type=bool, default=True, required=False,
                        help="Whether to use merging_layer."

***
Possible values of parameter 'sparse' are: 
'affinity', 'precomputed', and 'knn_neighbors_from_X'('affinity' means same as the dense matrix
'precomputed' means the input matrix)

Possible values of parameter 'affinity' are: 'cosine_similarity', 
'knn_neighbors_from_X', and 'precomputed' 
(for precomputed similarity matrix)

## Contact

Feel free to open an issue in Github or contact liangqs@tongji.edu.cn or qiusliang2-c@my.cityu.edu.hk if you have any problem in using TEC-O and TEC-U.

 
