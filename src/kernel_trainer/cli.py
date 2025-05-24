import sys
import click
import pickle
import datetime
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

import kernel_trainer.params as p
from kernel_trainer.main import kernel_generator
from kernel_trainer.preprocess import Preprocessor
from kernel_trainer.dataset import DataGenerator

logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=True,
    epilog="Specify one of these sub-commands and you can find more help from there.",
)
@click.pass_context
def cli(ctx, **kwargs):
    """
    Quantum Kernel Trainer
    """


@cli.command("train")
@p.file_path
@p.dimensions
@p.mode
@p.generations
@p.population
@p.chain_size
@p.mutpb
@p.cxpb
@p.processes
@p.out_path
def train(**kwargs):
    # Load CSV file
    file_path = kwargs.get("file_path")

    # Directory
    if file_path.is_dir():
        # List all CSV
        csv_files = list(file_path.glob("**/*.csv"))
        # Read each CSV file into a DataFrame and append to a list
        dataframes = [pd.read_csv(file) for file in csv_files]

        # Concatenate all DataFrames into a single DataFrame
        dataset = pd.concat(dataframes, ignore_index=True)
    else:
        dataset = pd.read_csv(file_path)

    num_dimensions = kwargs.get("dims")
    mode = kwargs.get("mode")
    prep = Preprocessor(num_dimensions, mode=mode, scale=True)
    X_reduced = prep.fit_transform(dataset.drop(columns=["y"]))

    # Split train and test
    X_train, _, y_train, _ = train_test_split(
        X_reduced, dataset["y"], test_size=0.20, random_state=42
    )

    logger.info(f"Training with {X_train.shape} dataset.")

    # Launch trainer
    config = {
        "X": X_train,
        "y": y_train,
        "population": kwargs.get("population"),
        "ngen": kwargs.get("generations"),
        "chain_size": kwargs.get("chain_size"),
        "mutpb": kwargs.get("mutpb"),
        "cxpb": kwargs.get("cxpb"),
        "processes": kwargs.get("processes"),
        "backend": kwargs.get("backend", "qiskit"),
    }

    pop_final, log, pops = kernel_generator(**config)

    results = {
        "input" : file_path,
        "kwargs": kwargs,
        "population": pop_final,
        "log": log,
        "population_registry": pops,
    }

    if "out_path" in kwargs:
        outpath = kwargs.get("out_path")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        with open(f"{outpath}_{mode}_{num_dimensions}_{timestamp}.pkl", "wb") as file:
            pickle.dump(results, file)


@cli.command("generate")
@p.out_path
@p.dataset_id
@p.samples
@p.imratio
@p.seed
def generate(**kwargs):
    # Dimensions
    num_dimensions = 3
    dataset_id = kwargs.get("dataset_id")
    samples = kwargs.get("samples")
    imbalance_ratio = kwargs.get("imb_ratio")
    seed = kwargs.get("seed")

    generator = DataGenerator(
        samples=samples, imbalance_ratio=imbalance_ratio, seed=seed
    )

    data = generator.generate_dataset(dataset_id)

    # Generate samples
    if "out_path" in kwargs:
        outpath = kwargs.get("out_path")
        if not outpath.exists():
            outpath.mkdir()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        data.to_csv(
            f"{outpath}/{dataset_id}_{num_dimensions}d_{samples}s_{imbalance_ratio}ir_{seed}_{timestamp}.csv"
        )


# Support running as a module
if __name__ == "__main__":
    cli()
