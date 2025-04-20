import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def main(args) -> None:
    """Combine chunk datasets into a single dataset."""
    datasets_path = Path(args.datasets)
    dataset_dirs = [
        d for d in datasets_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")
    ]

    datasets = [load_from_disk(str(dataset_dir)) for dataset_dir in dataset_dirs]

    combined_dataset = concatenate_datasets(datasets)

    combined_dataset.save_to_disk("{datasets_path}/combined")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract decoder outputs from Llama-2-7b-hf"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Path to the directory containing chunk datasets",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
