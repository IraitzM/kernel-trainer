# Kernel trainer

A clean version of the module used to find the most performant quantum kernel configuration. A simple Python library that performs required execution given a set of data. Data can also be generated according to what was described in the paper for 3-dimensional examples belonging to three different families of data (0, 1a,1b,1c,2a,2b,2c,3a,3b,3c).

Examples can be found under _notebooks_ folder.

## CLI usage

This library also allows for CLI access by specifying the corresponding parameters.

For data generation:
```py
ktrainer generate 
    --file-path file.csv
    --dataset-id 1a
    --samples 300
    --imbalance-ratio 0.1
```

Once the file is created, training can be triggered as follows:
```py
ktrainer train 
    --file-path file.csv
    --dims 3
    --mode pca
    --generations 100
    --population 1000
    --chain-size 20
    --mutpb 0.05
    --cxpb 0.5
    --processes 15 
    --out-path out
```
