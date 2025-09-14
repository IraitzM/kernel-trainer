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

Once the file is created, search of the best kernel can be triggered as follows:
```py
ktrainer search 
    --file-path file.csv
    --dims 3
    --mode raw
    --generations 100
    --population 1000
    --chain-size 20
    --mutpb 0.05
    --cxpb 0.5
    --processes 15 
    --out-path out
```
> [!TIP]
> [Taskfile](https://taskfile.dev/) exists with default parameters so you can simply call:
>
> $> task generate:1a
>
> $> task search:1a

Two other functionalities have been added to benchmark and compare results.

```py
ktrainer stats 
    --file-path file.csv
```

That takes the files that were obtained in the results folder and looks into the statistics for expressivity, entanglement capacity and CKA for each dataset id.

![](assets/stats.png)

The benchmark subcommand, takes an individual dataset id from its original dataset file and compares the obtained best individual against classical and pre-fixed quantum kernels.
```py
ktrainer benchmark 
    --file-path file.csv
    --dims 3
    --mode raw
```
A stats summary table will appear at the end for a particular dataset and best found individual labeled as `best`.

![](assets/benchmarkcli.png)