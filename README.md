# A Context-Enhanced Framework for Sequential Graph Reasoning
This is the official code for the models [CEF-GMPNN] and [CEF-RT] in the paper "A Context-Enhanced
 Framework for Sequential Graph Reasoning".
[CEF-GMPNN] and [CEF-RT] are new models that combine the CEF framework with two existing
 models. The CEF framework enhances the reasoning ability and stability of these two models
 in sequential graph tasks.
This project includes two folders, "CEF-GMPNN" and "CEF-RT", which correspond to the code
 implementations of [CEF-GMPNN] and [CEF-RT], respectively.

## Installation
We provide the environment files(`cef_gmpnn.yml` and `cef_rt.yml`) containing the packages used
 for these two models respectively. Assuming you have Anaconda installed (including conda), you
 can create a virtual environment using the environment file as follows:
1) Installation for the model [CEF-GMPNN]
```shell
conda env create -f cef_gmpnn.yml
conda activate algo3
```
2) Installation for the model [CEF-RT]
```shell
conda env create -f cef_rt.yml
conda activate rt_clrs
```

## Running Experiments
The internal structure of folders "CEF-GMPNN" and "CEF-RT" is the same, and the main files
 running are both in the "examples" directory, named "run.py". The file "run.py" in both
 "CEF-GMPNN" and "CEF-RT" contains the code we used to obtain the main results of our paper.
 You can reproduce our results by using the following commands.

1) Running CEF-GMPNN
When running CEF-GMPNN for the first time, you need to download the CLRS dataset from
 https://storage.googleapis.com/dm-clrs/CLRS30_v1.0.0.tar.gz and modify the dataset
 address to your download address in run.py. The modified code is as follows(on line 141
 of the "run.py" file. Alternatively, you can modify the default parameters while running
 the code).
```shell
flags.DEFINE_string('checkpoint_path', 'your dataset download path',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', 'your dataset download path',
                    'Path in which dataset is stored.')
```
After downloading the dataset, use the following command to train CEF-GMPNN to reproduce
 the results of the paper. You can modify the training algorithm by using the "algorithms"
 parameter.
```shell
cd CEF-GMPNN
python3 -m examples.run --algorithms mst_prim
```
We provide a script that can be run at once to obtain the single task results of the
 CEF-GMPNN model on all 30 algorithms. You can run it using the following command.
```shell
./run.sh
```
2) Running CEF-RT
The "run.py" file of CEF-RT can run all algorithms at once, and you can obtain all the main
 results of the CEF-RT model in the paper by using the following command. It should be noted
 that some algorithms require a large amount of memory space and require a higher storage
 space for running devices.
```shell
cd CEF-RT
python3 -m examples.run
```

## Some Details
* Run hyperparameters are set in `run.py` and can be overridden on the command line. You can
 obtain the desired result by modifying these hyperparameters.
* The current code of CEF-RT does not support chunking during training.