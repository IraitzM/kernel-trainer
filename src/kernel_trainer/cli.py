import os
import sys
import click
import pickle
import datetime
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

import tqdm
from rich.console import Console
from rich.table import Table

import kernel_trainer.params as p
from kernel_trainer.main import brute_force, kernel_generator
from kernel_trainer.kernels import (
    expsine2_kernel,
    sin_kernel,
    get_stats,
    get_matrices,
    get_matrices_ind,
)
from kernel_trainer.preprocess import Preprocessor
from kernel_trainer.dataset import DataGenerator

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Logs
logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level=os.getenv("LOG_LEVEL")
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


@cli.command("search")
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
@p.metric
@p.algo
@p.backend
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

    # Extract params
    algo = kwargs.get("algorithm")
    num_dimensions = kwargs.get("dims")
    mode = kwargs.get("mode")
    chain_size = kwargs.get("chain_size")

    # Preprocess
    prep = Preprocessor(num_dimensions, mode=mode, scale=True)
    X_reduced = prep.fit_transform(dataset.drop(columns=["y"]))

    # Split train and test
    X_train, _, y_train, _ = train_test_split(
        X_reduced, dataset["y"], test_size=0.20, random_state=42
    )
    logger.info(f"Training with {X_train.shape} dataset.")

    # Select algo
    if algo == "brute-force":
        logger.info(f"Going for the brute force approach for {4**chain_size} items")
        config = {
            "X": X_train[:, :num_dimensions],
            "y": y_train,
            "chain_size": chain_size,
            "processes": kwargs.get("processes"),
            "backend": kwargs.get("backend", "qiskit"),
            "metric": kwargs.get("metric", "CKA"),
        }
        pop_final, log = brute_force(**config)
    else:
        logger.info("Going for the evolutionary approach")
        config = {
            "X": X_train[:, :num_dimensions],
            "y": y_train,
            "num_pop": kwargs.get("population"),
            "ngen": kwargs.get("generations"),
            "chain_size": chain_size,
            "mutpb": kwargs.get("mutpb"),
            "cxpb": kwargs.get("cxpb"),
            "processes": kwargs.get("processes"),
            "backend": kwargs.get("backend", "qiskit"),
            "metric": kwargs.get("metric", "CKA"),
        }

        pop_final, log = kernel_generator(**config)

    # Result
    results = {
        "input": file_path,
        "kwargs": kwargs,
        "population": pop_final,
        "log": log,
    }

    outpath = kwargs.get("out_path")
    if outpath:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        with open(f"{outpath}_{mode}_{num_dimensions}_{timestamp}.pkl", "wb") as file:
            pickle.dump(results, file)


@cli.command("generate")
@p.out_path_man
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

        # Day precision
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        data.to_csv(
            f"{outpath}/{dataset_id}_{num_dimensions}d_{samples}s_{imbalance_ratio}ir_{seed}_{timestamp}.csv",
            index=False,
        )


@cli.command("stats")
@p.file_path
def stats(**kwargs):
    # Load result file
    file_path = kwargs.get("file_path")

    # Data files
    data = {
        "1a": [],
        "1b": [],
        "1c": [],
        "2a": [],
        "2b": [],
        "2c": [],
        "3a": [],
        "3b": [],
        "3c": [],
    }

    # Directory
    if file_path.is_dir():
        # List all files
        for x in os.listdir(file_path):
            if x.endswith(".pkl"):
                dataset = x[:2]  # First chars
                full_path = os.path.join(file_path, x)
                with open(full_path, "rb") as file:
                    tmp = pickle.load(file)

                data[dataset].append(tmp)

    # Summary table
    table = Table(title="Stats summary")
    table.add_column("Keys", style="cyan", no_wrap=True)
    table.add_column("Num experiments", style="magenta")
    table.add_column("Max. CKA", style="green")
    table.add_column("On experiment", style="green")
    table.add_column("Depth", style="green")
    table.add_column("Expresivity", style="green")
    table.add_column("Entganglement capacity", style="green")

    # Compute calculations
    keys = list(data.keys())

    for k in tqdm.tqdm(keys, desc="Iterating over keys"):
        num_exp = len(data[k])
        max_cka = 0.0
        max_id = 0
        individual = None
        nqubits = 0
        for idx, exp in enumerate(data[k]):
            if exp["log"][-1]["max"] > max_cka:
                max_cka = exp["log"][-1]["max"]
                max_id = idx
                individual = exp["population"][0]
                nqubits = exp["kwargs"]["dims"]

        # Best run
        depth, expr, entang = get_stats(individual, nqubits)

        table.add_row(
            str(k),
            str(num_exp),
            str(max_cka),
            str(max_id),
            str(depth),
            str(expr),
            str(entang),
        )

    console = Console()
    console.print(table)


@cli.command("benchmark")
@p.file_path
@p.dimensions
@p.mode
def benchmark(**kwargs):
    # Load dataset
    file_path = kwargs.get("file_path")

    # Look for data
    dataset = pd.read_csv(file_path)
    dataset_id = file_path.name[:2]
    logger.info(f"Dataset {dataset_id}")

    # Extract params
    num_dimensions = kwargs.get("dims")
    mode = kwargs.get("mode")

    # Preprocess
    prep = Preprocessor(num_dimensions, mode=mode, scale=True)
    X_reduced = prep.fit_transform(dataset.drop(columns=["y"]))

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, dataset["y"], test_size=0.20, random_state=42
    )
    logger.info(f"Training with {X_train.shape} dataset.")

    # Find best result
    max_cka = 0.0
    individual = None
    for x in os.listdir("./results/"):
        if x.endswith(".pkl") and x.startswith(dataset_id):
            full_path = os.path.join("./results/", x)
            with open(full_path, "rb") as file:
                experiment = pickle.load(file)

            if experiment["log"][-1]["max"] > max_cka:
                max_cka = experiment["log"][-1]["max"]
                individual = experiment["population"][0]
    logger.info(f"Max CKA registered {max_cka}")

    # Summary table
    table = Table(title="Stats summary")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("AUC", style="magenta")
    table.add_column("F1", style="blue")
    table.add_column("CKA", style="green")

    # Classical
    for svc_type in ["linear", "poly", "rbf", "sin", "expsine"]:
        if svc_type == "expsine":
            model = SVC(kernel=expsine2_kernel, probability=True, random_state=42)
        elif svc_type == "sin":
            model = SVC(kernel=sin_kernel, probability=True, random_state=42)
        else:
            model = SVC(kernel=svc_type, probability=True, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred)
        f1score = f1_score(y_test, model.predict(X_test))

        table.add_row(svc_type, str(roc_auc), str(f1score), "--")

    # Quantum
    for qsvc in ["Z", "ZZ"]:
        m_train, m_test, cka = get_matrices(X_train, X_test, y_train, qsvc)

        model = SVC(kernel="precomputed", probability=True, random_state=42)
        model.fit(m_train, y_train)

        y_pred = model.predict_proba(m_test)[:, 1]
        roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred)
        f1score = f1_score(y_test, model.predict(m_test))

        table.add_row(qsvc, str(roc_auc), str(f1score), str(cka))

    # Best quantum
    m_train, m_test, cka = get_matrices_ind(X_train, X_test, y_train, individual)

    model = SVC(kernel="precomputed", probability=True, random_state=42)
    model.fit(m_train, y_train)

    y_pred = model.predict_proba(m_test)[:, 1]
    roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    f1score = f1_score(y_test, model.predict(m_test))

    table.add_row("best", str(roc_auc), str(f1score), str(cka))

    console = Console()
    console.print(table)


# Support running as a module
if __name__ == "__main__":
    cli()
