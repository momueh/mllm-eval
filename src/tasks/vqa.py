import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Callable

logger = logging.getLogger("vqa_task")


def load_vqa_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load VQA dataset from a JSON file."""
    with open(dataset_path, "r") as f:
        return json.load(f)


def get_image_path(image_id: int, images_dir: str) -> str:
    """Get the image path from the image ID.

    VQA dataset typically uses COCO image IDs, which follow a specific filename pattern
    """
    # COCO val2014 images are typically named: COCO_val2014_000000123456.jpg
    # where 123456 is the image_id padded to 12 digits
    image_filename = f"COCO_val2014_{image_id:012d}.jpg"
    return os.path.join(images_dir, image_filename)


def get_random_subset(
    questions: List[Dict[str, Any]], subset_size: int
) -> List[Dict[str, Any]]:
    """Get a random subset of questions."""
    # Set a seed for reproducibility
    random.seed(42)

    if subset_size >= len(questions):
        return questions

    return random.sample(questions, subset_size)


def run_vqa_task(model_manager, subset_size: int, results_dir: str):
    """Run VQA task for all three models on the same subset of data."""
    logger.info("Starting VQA task")

    # Load questions dataset
    questions_path = "src/tasks/data/vqa/v2_OpenEnded_mscoco_val2014_questions.json"  # Adjust path as needed
    images_dir = "src/tasks/data/vqa/val2014"  # Adjust path as needed - this should point to the COCO val2014 images

    dataset = load_vqa_dataset(questions_path)
    questions = dataset["questions"]
    logger.info(f"Loaded dataset with {len(questions)} questions")

    # Get random subset
    subset = get_random_subset(questions, subset_size)
    logger.info(f"Selected random subset of {len(subset)} questions")

    # Initialize results structure for each model
    results = {"openai": [], "anthropic": [], "gemini": []}

    # Process each question in the subset
    for question_item in subset:
        question_id = question_item["question_id"]
        image_id = question_item["image_id"]
        question = question_item["question"]

        # Get image path from image ID
        try:
            image_path = get_image_path(image_id, images_dir)

            # Check if image exists
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                error_message = f"Image not found for image_id: {image_id}"
                # Add error entry for all models
                for model in results:
                    results[model].append(
                        {"question_id": question_id, "answer": error_message}
                    )
                continue

            # Create prompt
            prompt = f"Have a look at the image and answer the question in one single word. Use only lowercase letters and no punctuation: {question}"

            logger.info(f"Processing question ID {question_id} (image ID: {image_id})")

            # Query all models at once
            model_responses = model_manager.query_all_models(prompt, image_path)

            # Store responses in results
            for model, response in model_responses.items():
                results[model].append({"question_id": question_id, "answer": response})

        except Exception as e:
            logger.error(f"Error processing question ID {question_id}: {str(e)}")
            error_message = f"Error: {str(e)}"
            # Add error entry for all models
            for model in results:
                results[model].append(
                    {"question_id": question_id, "answer": error_message}
                )

    # Save results for each model
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    for model, model_results in results.items():
        output_file = results_path / f"vqa_results_{model}.json"
        with open(output_file, "w") as f:
            json.dump(model_results, f, indent=2)

        logger.info(f"Saved {model} results to {output_file}")
