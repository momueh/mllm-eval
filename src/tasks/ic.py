import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any
import evaluate
from datasets import load_dataset

logger = logging.getLogger("image_captioning_task")


def load_flickr_dataset(split: str = "test", sample_size: int = 100):
    """Load flickr30k dataset from HuggingFace with a limited number of examples."""
    logger.info(
        f"Loading lmms-lab/flickr30k dataset, {split} split (limited to {sample_size} examples)..."
    )

    # Option 1: Load only a portion of the split directly
    dataset = load_dataset("lmms-lab/flickr30k", split=f"{split}[:{sample_size}]")

    # Option 2 (alternative): For random sampling instead of first N examples
    # full_dataset = load_dataset("HuggingFaceM4/COCO", split=split)
    # sampled_indices = random.sample(range(len(full_dataset)), min(sample_size, len(full_dataset)))
    # dataset = full_dataset.select(sampled_indices)

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def get_image_caption_prompt(example: Dict[str, Any]) -> str:
    """Create a prompt for the image captioning task."""
    return "Create a caption for this image. Respond with a single sentence."


def evaluate_captions(predictions: List[str], references: List[List[str]]):
    """
    Evaluate the predicted captions against reference captions using BLEU and ROUGE.

    Args:
        predictions: List of predicted captions
        references: List of lists of reference captions

    Returns:
        Dictionary with evaluation metrics
    """
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    # Compute metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    # Combine metrics
    metrics = {
        "bleu": bleu_score,
        "rouge": rouge_score,
    }

    return metrics


def run_image_captioning_task(model_manager, subset_size: int, results_dir: str):
    """Run image captioning task for all models on the same subset of data."""
    logger.info("Starting image captioning task")

    # Load Flickr  dataset
    dataset = load_flickr_dataset(split="test", sample_size=subset_size)

    # Initialize results structure for each model
    results = {"openai": [], "anthropic": [], "gemini": []}

    # Process each image in the dataset
    for i, example in enumerate(dataset):
        image_id = example.get("img_id", str(i))
        image = example["image"]
        captions = example["caption"]  # List of captions for this image

        # Create prompt
        prompt = get_image_caption_prompt(example)

        logger.info(f"Processing image {i+1}/{len(dataset)} (ID: {image_id})")

        try:
            # Query all models at once
            model_responses = model_manager.query_all_models(prompt, image)

            # Store responses and reference captions in results
            for model, response in model_responses.items():
                results[model].append(
                    {
                        "image_id": image_id,
                        "prompt": prompt,
                        "response": response,
                        "references": captions,  # Store all reference captions
                    }
                )

        except Exception as e:
            logger.error(f"Error processing image {image_id}: {str(e)}")
            error_message = f"Error: {str(e)}"

            # Add error entry for all models
            for model in results:
                results[model].append(
                    {
                        "image_id": image_id,
                        "prompt": prompt,
                        "response": error_message,
                        "references": captions,
                    }
                )

    # Create results directory path
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    # Calculate metrics and save results for each model
    metrics = {}

    for model, model_results in results.items():
        # Extract predictions and references for evaluation
        predictions = [
            item["response"]
            for item in model_results
            if not item["response"].startswith("Error:")
        ]
        references = [
            item["references"]
            for item in model_results
            if not item["response"].startswith("Error:")
        ]

        # Calculate metrics
        model_metrics = evaluate_captions(predictions, references)
        metrics[model] = model_metrics

        # Save results
        output_file = results_path / f"image_captioning_results_{model}.json"
        with open(output_file, "w") as f:
            json.dump(model_results, f, indent=2)

        logger.info(f"Saved {model} results to {output_file}")

    # Save aggregated metrics
    metrics_file = results_path / "image_captioning_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {metrics_file}")

    return metrics
