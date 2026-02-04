import os
import click
import pickle
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

import tqdm
from rich.console import Console
from rich.table import Table
from rich_tools import table_to_df

import kernel_trainer.params as p
from kernel_trainer.config import logger
from kernel_trainer.main import brute_force, kernel_generator
from kernel_trainer.kernels import (
    expsine2_kernel,
    sin_kernel,
    get_stats,
    get_matrices,
    get_scores_ind,
)
from kernel_trainer.preprocess import Preprocessor
from kernel_trainer.dataset import DataGenerator

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score

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
@p.dataset
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
@p.seed
def train(**kwargs):
    """
    CLI command to search for optimal feature maps.

    Parameters
    ----------
    **kwargs
        Keyword arguments populated from click options. Expected keys include
        ``dataset``, ``dims``, ``mode``, ``generations``, ``population``,
        ``chain_size``, ``mutpb``, ``cxpb``, ``processes``, ``out_path``,
        ``metric``, ``algo``, and ``backend``.

    Returns
    -------
    None
    """
    # Load data
    dataset = kwargs.get("dataset")
    num_dimensions = kwargs.get("dims")
    mode = kwargs.get("mode")
    seed = kwargs.get("seed")

    if not isinstance(dataset, str):
        # Directory
        if dataset.is_dir():
            # List all CSV
            csv_files = list(dataset.glob("**/*.csv"))
            # Read each CSV file into a DataFrame and append to a list
            dataframes = [pd.read_csv(file) for file in csv_files]
            # Concatenate all DataFrames into a single DataFrame
            data = pd.concat(dataframes, ignore_index=True)
        else:
            data = pd.read_csv(dataset)

        # Preprocess
        prep = Preprocessor(num_dimensions, mode=mode, scale=True)
        X_reduced = prep.fit_transform(data.drop(columns=["y"]))

        # Split train and test
        X_train, _, y_train, _ = train_test_split(
            X_reduced, data["y"], test_size=0.20, random_state=seed
        )
        logger.info(f"Training with {X_train.shape} dataset.")
    else:
        if dataset == "iris":
            from sklearn.datasets import load_iris

            # Load data
            df_iris = load_iris(as_frame=True)
            X_df = df_iris["data"]
            y_df = df_iris["target"]

            # Filter two most complex classes
            X_masked = (X_df[y_df > 0]).to_numpy()
            y_masked = (y_df[y_df > 0]).to_numpy()

            _, num_dimensions = X_masked.shape

            # Preprocess
            prep = Preprocessor(num_dimensions, mode=mode, scale=True)
            X_reduced = prep.fit_transform(X_masked)

            # Split train and test
            X_train, _, y_train, _ = train_test_split(
                X_reduced, y_masked, test_size=0.20, random_state=seed
            )
            logger.info(f"Training with {X_train.shape} dataset.")
        elif dataset == "wine":
            from sklearn.datasets import load_wine

            # Load data
            df_iris = load_wine(as_frame=True)
            X_df = df_iris["data"]
            y_df = df_iris["target"]

            # Filter two most complex classes
            X_masked = (X_df[y_df > 0]).to_numpy()
            y_masked = (y_df[y_df > 0]).to_numpy()

            _, num_dimensions = X_masked.shape

            # Preprocess
            prep = Preprocessor(num_dimensions, mode=mode, scale=True)
            X_reduced = prep.fit_transform(X_masked)

            # Split train and test
            X_train, _, y_train, _ = train_test_split(
                X_reduced, y_masked, test_size=0.20, random_state=seed
            )
            logger.info(f"Training with {X_train.shape} dataset.")
        elif dataset == "monk":
            logger.error(f"Dataset {dataset} not yet implemented")
            raise NotImplementedError
        else:
            logger.error(f"Dataset {dataset} not found!")
            raise ValueError(f"Dataset '{dataset}' not found. Supported: iris, wine")

    # Extract params
    algo = kwargs.get("algorithm")
    chain_size = kwargs.get("chain_size")

    # Check remainder
    rem = chain_size % X_train.shape[1]
    if rem > 0:
        logger.error(
            f"Dataset is {X_train.shape} but your chain-size is {chain_size}, try multiples of dataset width"
        )
        raise Exception()

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
        "input": dataset,
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
    """
    Generate and save a synthetic dataset based on pre-defined templates.

    Parameters
    ----------
    **kwargs
        Keyword arguments from click options, including ``dataset_id``,
        ``samples``, ``imb_ratio`` and ``seed``. If ``out_path`` is provided,
        generated CSV files will be saved there.

    Returns
    -------
    None
    """
    # Dimensions
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
            f"{outpath}/{dataset_id}_3d_{samples}s_{imbalance_ratio}ir_{seed}_{timestamp}.csv",
            index=False,
        )


@cli.command("stats")
@p.file_path
@p.id
@p.out_path
def stats(**kwargs):
    """
    Aggregate and present statistics from stored experiment pickles.

    Parameters
    ----------
    **kwargs
        Keyword arguments include ``file_path`` (file or directory to scan) and
        optional ``id`` to filter experiments.

    Returns
    -------
    None
    """
    # Load result file
    file_path = kwargs.get("file_path")

    identity = kwargs.get("id", None)

    if identity:
        data = {identity: []}

        if file_path.is_file():
            with open(file_path, "rb") as file:
                tmp = pickle.load(file)
            data[identity].append(tmp)
        # Directory
        elif file_path.is_dir():
            # List all files
            for x in os.listdir(file_path):
                if x.endswith(".pkl") and identity in x:
                    full_path = os.path.join(file_path, x)
                    with open(full_path, "rb") as file:
                        tmp = pickle.load(file)

                    data[identity].append(tmp)
    else:
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
            dataset = file_path.name[:2]
            if dataset in data:
                with open(file_path, "rb") as file:
                    tmp = pickle.load(file)
                data[dataset].append(tmp)
        elif file_path.is_dir():
            for x in os.listdir(file_path):
                if x.endswith(".pkl"):
                    dataset = x[:2]  # First chars
                    if dataset in data:
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
    table.add_column("Expressivity", style="green")
    table.add_column("Entanglement capacity", style="green")

    # Compute calculations
    keys = list(data.keys())

    for k in tqdm.tqdm(keys, desc="Iterating over keys"):
        num_exp = len(data[k])
        max_cka = 0.0
        max_id = 0
        individual = None
        nqubits = 0
        for idx, exp in enumerate(data[k]):
            # Guard: ensure log exists and is non-empty before indexing
            log = exp.get("log")
            if not log:
                continue

            # Safely read last log entry and its max value
            last_entry = log[-1] if isinstance(log, list) else {}
            max_val = last_entry.get("max", 0.0)

            if max_val > max_cka:
                max_cka = max_val
                max_id = idx
                population = exp.get("population") or []
                individual = population[0] if population else None
                nqubits = exp.get("kwargs", {}).get("dims", 0) or 0

        # If no valid individual was found, skip this key
        if individual is None or nqubits <= 0:
            logger.warning(f"No valid experiments with non-empty logs found for key {k}; skipping.")
            continue

        # Best run
        depth, expr, entang = get_stats(individual, nqubits)

        table.add_row(
            str(k),
            str(num_exp),
            str(max_cka),
            str(max_id),
            str(depth),
            str(round(expr, 6)),
            str(round(entang, 6)),
        )

    outpath = kwargs.get("out_path", None)
    if outpath:
        if not outpath.exists():
            outpath.mkdir()

        # Day precision
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        data = table_to_df(table)

        data.to_csv(
            f"{outpath}/stats_{timestamp}.csv",
            index=False,
        )
    else:
        console = Console()
        console.print(table)


@cli.command("benchmark")
@p.dataset
@p.dimensions
@p.mode
@p.seed
@p.file_path
@p.out_path
def benchmark(**kwargs):
    """
    Run benchmark procedures on a dataset and export results to CSV.

    Parameters
    ----------
    **kwargs
        Keyword arguments from click options such as ``dataset``, ``dims``,
        ``mode``, ``seed`` and ``out_path``.

    Returns
    -------
    None
    """
    # Load dataset
    dataset = kwargs.get("dataset")
    num_dimensions = kwargs.get("dims")
    mode = kwargs.get("mode", "raw")
    seed = kwargs.get("seed")

    dataset_id = None
    if not isinstance(dataset, str):
        # Look for data
        dataset_id = dataset.name[:2]

        # Directory
        if dataset.is_dir():
            # List all CSV
            csv_files = list(dataset.glob("**/*.csv"))
            # Read each CSV file into a DataFrame and append to a list
            dataframes = [pd.read_csv(file) for file in csv_files]
            # Concatenate all DataFrames into a single DataFrame
            data = pd.concat(dataframes, ignore_index=True)
        else:
            data = pd.read_csv(dataset)

        # Preprocess
        prep = Preprocessor(num_dimensions, mode=mode, scale=True)
        X_reduced = prep.fit_transform(data.drop(columns=["y"]))

        # Split train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, data["y"], test_size=0.20, random_state=seed
        )
    else:
        # Id
        dataset_id = dataset

        if dataset == "iris":
            from sklearn.datasets import load_iris

            # Load data
            df_iris = load_iris(as_frame=True)
            X_df = df_iris["data"]
            y_df = df_iris["target"]

            # Filter two most complex classes
            X_masked = (X_df[y_df > 0]).to_numpy()
            y_masked = (y_df[y_df > 0]).to_numpy()

            _, num_dimensions = X_masked.shape

            # Preprocess
            prep = Preprocessor(num_dimensions, mode=mode, scale=True)
            X_reduced = prep.fit_transform(X_masked)

            # Split train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X_reduced, y_masked, test_size=0.20, random_state=seed
            )
        elif dataset == "wine":
            from sklearn.datasets import load_wine

            # Load data
            df_iris = load_wine(as_frame=True)
            X_df = df_iris["data"]
            y_df = df_iris["target"]

            # Filter two most complex classes
            X_masked = (X_df[y_df > 0]).to_numpy()
            y_masked = (y_df[y_df > 0]).to_numpy()

            _, num_dimensions = X_masked.shape

            # Preprocess
            prep = Preprocessor(num_dimensions, mode=mode, scale=True)
            X_reduced = prep.fit_transform(X_masked)

            # Split train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X_reduced, y_masked, test_size=0.20, random_state=seed
            )
        elif dataset == "monk":
            logger.error(f"Dataset {dataset} not yet implemented")
            raise NotImplementedError
        else:
            logger.error(f"Dataset {dataset} not found!")
            raise ValueError(f"Dataset '{dataset}' not found. Supported: iris, wine")

    # Log
    logger.info(f"Dataset {dataset}")
    logger.info(f"Training with {X_train.shape} dataset.")

    # Find best result
    max_cka = 0.0
    individual = None

    # Load result file
    file_path = kwargs.get("file_path")
    if not file_path or not file_path.is_dir():
        logger.error(f"file_path must be an existing directory, got: {file_path}")
        raise ValueError("Invalid file_path for benchmark results")

    for x in os.listdir(file_path):
        if x.endswith(".pkl") and x.startswith(dataset_id):
            full_path = os.path.join(file_path, x)
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
        logger.debug(f"Running {svc_type} training")
        if svc_type == "expsine":
            model = SVC(kernel=expsine2_kernel, probability=True, random_state=seed)
        elif svc_type == "sin":
            model = SVC(kernel=sin_kernel, probability=True, random_state=seed)
        else:
            model = SVC(kernel=svc_type, probability=True, random_state=seed)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred)
        f1score = f1_score(y_test, model.predict(X_test))

        table.add_row(svc_type, str(roc_auc), str(f1score), "--")

    # Quantum
    for qsvc in ["Z", "ZZ-full", "ZY", "ZZ-linear", "ZY-linear", "XY"]:
        logger.debug(f"Running {qsvc} QSVM training")
        m_train, m_test, cka = get_matrices(X_train, X_test, y_train, qsvc)

        model = SVC(kernel="precomputed", probability=True, random_state=seed)
        model.fit(m_train, y_train)

        y_pred = model.predict_proba(m_test)[:, 1]
        roc_auc = roc_auc_score(y_true=y_test, y_score=y_pred)
        f1score = f1_score(y_test, model.predict(m_test))

        table.add_row(qsvc, str(roc_auc), str(f1score), str(cka))

    # Best quantum
    # Qiskit - NOT WORKING
    # roc_auc, f1score, cka = get_scores_ind(X_train, X_test, y_train, y_test, individual)
    # table.add_row("best (qiskit)", str(roc_auc), str(f1score), str(cka))

    # Pennylane
    roc_auc, f1score, cka = get_scores_ind(
        X_train, X_test, y_train, y_test, individual, backend="pennylane", seed=seed
    )
    table.add_row("best (pennylane)", str(roc_auc), str(f1score), str(cka))
    if cka < max_cka:
        logger.warning(f"CKA in this execution is lower than the original max CKA: {cka} < {max_cka}")
    elif cka > max_cka:
        logger.warning(f"CKA in this execution is higher than the original max CKA: {cka} > {max_cka}")

    out_path = kwargs.get("out_path", None)
    if out_path:
        outpath = out_path / "benchmark_results"
        if not outpath.exists():
            outpath.mkdir()

        # Day precision
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        data = table_to_df(table)

        data.to_csv(
            f"{outpath}/benchmark_{dataset_id}_{num_dimensions}d_{seed}_{timestamp}.csv",
            index=False,
        )
    else:
        console = Console()
        console.print(table)

@cli.command("compact")
@p.dataset
@p.out_path_man
def compact(**kwargs):
    """
    Compact results from the benchmark

    Parameters
    ----------
    **kwargs
        Keyword arguments from click options such as ``dataset``, ``dims``,
        ``mode``, ``seed`` and ``out_path``.

    Returns
    -------
    None
    """
    # Load dataset
    dataset_id = kwargs.get("dataset")
    dataset_path = kwargs.get("out_path")

    # List all CSV
    csv_files = list(dataset_path.glob("**/*.csv"))

    # Read all CSV files
    dataframes = []
    for csv_file in csv_files:
        if dataset_id in csv_file.name:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                logger.info(f"Loaded {csv_file}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue

    # Check that all dataframes have the same structure
    first_columns = dataframes[0].columns.tolist()
    first_methods = dataframes[0].iloc[:, 0].tolist()

    for i, df in enumerate(dataframes[1:], 1):
        if df.columns.tolist() != first_columns:
            logger.warning(f"File {csv_files[i]} has different columns")
        if df.iloc[:, 0].tolist() != first_methods:
            logger.warning(f"File {csv_files[i]} has different methods/rows")

    # Create a result dataframe with the same structure
    result_df = dataframes[0].copy()

    # For each numeric column (skip the first column which is the Method name)
    for col in result_df.columns[1:]:
        # Collect all values from all dataframes for this column
        all_values = []
        for df in dataframes:
            # Convert to numeric, replacing '--' and other non-numeric values with NaN
            values = pd.to_numeric(df[col], errors='coerce')
            all_values.append(values)

        # Stack all values and compute mean (ignoring NaN)
        stacked = pd.concat(all_values, axis=1)
        means = stacked.mean(axis=1)

        # Update result dataframe
        # If all values were NaN, keep as '--', otherwise show the mean
        result_df[col] = means.apply(lambda x: '--' if pd.isna(x) else f"{x:.6f}")

    # Day precision
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    output_file = f"{dataset_path}/benchmark_{dataset_id}_compact_{timestamp}.csv"

    # Save to output file
    result_df.to_csv(output_file, index=False)
    logger.info(f"\nMean results saved to: {output_file}")

# Support running as a module
if __name__ == "__main__":
    cli()
