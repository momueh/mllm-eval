import os
import json
import logging
import argparse
from pathlib import Path

from src.utils.api_manager import APIManager
from src.utils.model_manager import ModelManager
from src.tasks.vqa import run_vqa_task
from src.tasks.ic import run_image_captioning_task
from src.tasks.vcr import run_vcr_task
from src.tasks.visdial import run_visdial_task


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("main")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multimodal LLM Evaluation")
    parser.add_argument(
        "--task",
        type=str,
        choices=["vqa", "ic", "vcr", "visdial"],
        required=True,
        help="Task to evaluate",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=30,
        help="Number of examples to use for the task",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/api_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save results"
    )
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()
    logger.info(
        f"Starting evaluation on {args.task} task with subset size {args.subset_size}"
    )

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Create results directory if it doesn't exist
    results_path = Path(args.results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    # Initialize API manager and model manager
    api_manager = APIManager(config)
    model_manager = ModelManager(config, api_manager)

    # Run the specified task
    if args.task == "vqa":
        run_vqa_task(
            model_manager=model_manager,
            subset_size=args.subset_size,
            results_dir=args.results_dir,
        )
    elif args.task == "ic":
        run_image_captioning_task(
            model_manager=model_manager,
            subset_size=args.subset_size,
            results_dir=args.results_dir,
        )
    elif args.task == "vcr":
        vcr_data_dir = "src/tasks/data/vcr"
        vcr_split = "val"

        run_vcr_task(
            model_manager=model_manager,
            data_dir=vcr_data_dir,
            subset_size=args.subset_size,
            results_dir=args.results_dir,
            split=vcr_split,
        )
    elif args.task == "visdial":
        visdial_data_dir = "src/tasks/data/visdial"
        visdial_split = "val"

        run_visdial_task(
            model_manager=model_manager,
            data_dir=visdial_data_dir,
            subset_size=args.subset_size,
            results_dir=args.results_dir,
            split=visdial_split,
        )

    logger.info(f"Task {args.task} completed. Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
