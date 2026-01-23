import click
from pathlib import Path

file_path = click.option(
    "--file-path",
    envvar=None,
    help="CSV file path to load the dataset from.",
    type=click.Path(resolve_path=True, path_type=Path),
    prompt=True,
)

dimensions = click.option(
    "--dims",
    envvar=None,
    help="Dimensions to reduce it to.",
    default=2,
    type=click.INT,
    prompt=True,
)

mode = click.option(
    "--mode",
    envvar=None,
    help="Dimensionality reduction mode.",
    default="pca",
    type=click.Choice(["lda", "pca", "raw", "tsne"]),
    prompt=True,
)

backend = click.option(
    "--backend",
    envvar=None,
    help="Backend to be used",
    default="qiskit",
    type=click.Choice(["qiskit", "pennylane"]),
)

generations = click.option(
    "--generations",
    envvar=None,
    help="Number of generations.",
    default=2,
    type=click.INT,
)

population = click.option(
    "--population",
    envvar=None,
    help="Number of individuals.",
    default=2,
    type=click.INT,
)

chain_size = click.option(
    "--chain-size",
    envvar=None,
    help="Size of the FM chain",
    default=2,
    type=click.INT,
    prompt=True,
)

mutpb = click.option(
    "--mutpb",
    envvar=None,
    help="Probability of mutation",
    default=0.05,
    type=click.FLOAT,
)

cxpb = click.option(
    "--cxpb",
    envvar=None,
    help="Probability of crossbreed",
    default=0.5,
    type=click.FLOAT,
)

processes = click.option(
    "--processes",
    envvar=None,
    help="Number of cores to run on.",
    default=2,
    type=click.INT,
)

out_path = click.option(
    "--out-path",
    envvar=None,
    help="Output folder/file",
    type=click.Path(resolve_path=True, path_type=Path),
    default=None,  # "results/train",
)

out_path_man = click.option(
    "--out-path",
    envvar=None,
    help="Output folder/file",
    type=click.Path(resolve_path=True, path_type=Path),
    prompt=True,
)

dataset_id = click.option(
    "--dataset-id",
    envvar=None,
    help="Dataset ID",
    default="0",
    type=click.Choice(["0", "1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c"]),
    prompt=True,
)

metric = click.option(
    "--metric",
    envvar=None,
    help="Metric to be used",
    default="CKA",
    type=click.Choice(["KTA", "CKA"]),
)

algo = click.option(
    "--algorithm",
    envvar=None,
    help="Algorithm to be used",
    default="evolutionary",
    type=click.Choice(["brute-force", "evolutionary"]),
)

samples = click.option(
    "--samples",
    envvar=None,
    help="Number of samples to generate",
    default=100,
    type=click.INT,
    prompt=True,
)

imratio = click.option(
    "--imb-ratio",
    envvar=None,
    help="Ratio between samples",
    default=0.5,
    type=click.FLOAT,
    prompt=True,
)

seed = click.option(
    "--seed",
    envvar=None,
    help="Seed",
    default=4321,
    type=click.INT,
)

cache = click.option(
    "--cache",
    is_flag=True,
    help="To enable cache",
    default=False,
)
