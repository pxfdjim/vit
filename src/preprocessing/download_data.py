"""Download Orion benchmark series with verified, atomic writes."""
import argparse
import ast
import csv
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BASE_URL = "https://sintel-orion.s3.us-east-2.amazonaws.com"
YAHOO_DIRS = {
    "YAHOOA1": "A1Benchmark",
    "YAHOOA2": "A2Benchmark",
    "YAHOOA3": "A3Benchmark",
    "YAHOOA4": "A4Benchmark",
}


def new_session():
    session = requests.Session()
    session.trust_env = False  # The host's HTTP proxy cannot reach this bucket.
    return session


def download_file(name, destination, force=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{name}"
    with new_session() as session:
        if destination.is_file() and not force:
            try:
                head = session.head(url, timeout=(10, 30))
                head.raise_for_status()
                expected = int(head.headers.get("Content-Length", -1))
                if expected > 0 and destination.stat().st_size == expected:
                    return "existing"
            except requests.RequestException:
                pass

        last_error = None
        for attempt in range(3):
            temporary = destination.with_name(destination.name + ".part")
            try:
                with session.get(url, stream=True, timeout=(10, 90)) as response:
                    response.raise_for_status()
                    expected = int(response.headers.get("Content-Length", -1))
                    received = 0
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                                received += len(chunk)
                    if expected >= 0 and received != expected:
                        raise IOError(f"expected {expected} bytes, received {received}")
                os.replace(temporary, destination)
                return "downloaded"
            except (requests.RequestException, OSError) as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(attempt + 1)
        raise RuntimeError(f"{name}: {last_error}")


def verify_yahoo_file(dataset, signal, ground_truth):
    """Validate a locally supplied Yahoo S5 series against Orion ground truth."""
    source = DATA_DIR / "Yahoo_S5" / YAHOO_DIRS[dataset] / f"{signal}.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Yahoo S5 source missing: {source}")
    label_column = "is_anomaly" if dataset in ("YAHOOA1", "YAHOOA2") else "anomaly"
    time_column = "timestamp" if dataset in ("YAHOOA1", "YAHOOA2") else "timestamps"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {time_column, "value", label_column}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"{source}: expected columns {sorted(needed)}")
        observed = set()
        timestamps = set()
        count = 0
        for count, row in enumerate(reader, start=1):
            timestamp = row[time_column]
            if timestamp in timestamps:
                raise ValueError(f"{source}: duplicate timestamp {timestamp}")
            timestamps.add(timestamp)
            if not math.isfinite(float(row["value"])):
                raise ValueError(f"{source}: non-finite value on row {count}")
            label = int(row[label_column])
            if label not in (0, 1):
                raise ValueError(f"{source}: invalid label {label} on row {count}")
            if label:
                observed.add(count)
    if not count:
        raise ValueError(f"{source}: empty series")
    if signal not in ground_truth:
        raise ValueError(f"{signal}: ground truth missing from anomalies.csv")
    expected = {point for start, end in ground_truth[signal]
                for point in range(int(start), int(end) + 1)}
    if observed != expected:
        raise ValueError(f"{signal}: labels differ from anomalies.csv "
                         f"({len(observed ^ expected)} points)")
    return "verified"


def main():
    parser = argparse.ArgumentParser(description="Download Orion benchmark data.")
    parser.add_argument("datasets", nargs="*", help="Dataset names; default is all names in datasets.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    for name in ("datasets.csv", "anomalies.csv"):
        state = download_file(name, DATA_DIR / name, args.force)
        print(f"{name}: {state}", flush=True)

    with (DATA_DIR / "datasets.csv").open(newline="") as source:
        index = {row[0]: ast.literal_eval(row[1]) for row in csv.reader(source) if len(row) == 2}
    selected = args.datasets or list(index)
    unknown = sorted(set(selected) - set(index))
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")

    with (DATA_DIR / "anomalies.csv").open(newline="") as source:
        next(source)
        ground_truth = {signal: ast.literal_eval(events)
                        for signal, events in csv.reader(source)}

    tasks = [(dataset, signal) for dataset in selected for signal in index[dataset]]
    print(f"Checking {len(tasks)} signals from {len(selected)} datasets", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            (pool.submit(verify_yahoo_file, dataset, signal, ground_truth)
             if dataset in YAHOO_DIRS else
             pool.submit(download_file, f"{signal}.csv", DATA_DIR / dataset / f"{signal}.csv", args.force)): (dataset, signal)
            for dataset, signal in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            dataset, signal = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{dataset}/{signal}: {error}")

    if failures:
        for failure in failures:
            print(f"FAILED {failure}", flush=True)
        raise SystemExit(f"{len(failures)} of {len(tasks)} downloads failed")
    print(f"All {len(tasks)} signals verified", flush=True)


if __name__ == "__main__":
    main()
