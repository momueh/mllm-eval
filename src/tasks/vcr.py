import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger("visual_commonsense_reasoning_task")


def detokenize_text(tokenized_text, objects):
    """
    Convert tokenized text with object references to natural language string.

    Args:
        tokenized_text: List of tokens, where object references are sub-lists of indices
        objects: List of object types (e.g., ["person", "person", "horse"])

    Returns:
        String with object references converted to readable text
    """
    result = []
    for token in tokenized_text:
        if isinstance(token, list):
            # This is an object reference
            obj_indices = token
            if len(obj_indices) == 1:
                # Single object reference
                obj_idx = obj_indices[0]
                if obj_idx < len(objects):
                    result.append(f"[{objects[obj_idx]}_{obj_idx}]")
                else:
                    # Index out of range, use a placeholder
                    result.append(f"[object_{obj_idx}]")
            else:
                # Multiple object references
                obj_names = []
                for idx in obj_indices:
                    if idx < len(objects):
                        obj_names.append(f"[{objects[idx]}_{idx}]")
                    else:
                        obj_names.append(f"[object_{idx}]")
                result.append(" and ".join(obj_names))
        else:
            # Normal text token
            result.append(token)
    return " ".join(result)


def load_vcr_dataset(
    data_dir: str, split: str = "val", sample_size: Optional[int] = None
):
    """
    Load VCR dataset from local files.

    Args:
        data_dir: Path to the VCR dataset directory
        split: Dataset split ("train", "val", or "test")
        sample_size: Maximum number of examples to load (None for all)

    Returns:
        List of examples with image paths
    """
    logger.info(
        f"Loading VCR dataset from {data_dir}, {split} split"
        + (f" (limited to {sample_size} examples)" if sample_size else "")
    )

    # Construct path to the annotations file
    annot_file = os.path.join(data_dir, f"{split}.jsonl")

    if not os.path.exists(annot_file):
        raise FileNotFoundError(f"VCR annotations file not found at {annot_file}")

    examples = []
    skipped = 0

    with open(annot_file, "r") as f:
        for i, line in enumerate(f):
            if sample_size is not None and len(examples) >= sample_size:
                break

            try:
                example = json.loads(line.strip())

                # Add the image path
                img_fn = example["img_fn"]
                img_path = os.path.join(data_dir, "vcr1images", img_fn)

                if not os.path.exists(img_path):
                    logger.warning(f"Image file not found: {img_path}")
                    skipped += 1
                    continue

                example["image_path"] = img_path
                examples.append(example)

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line {i} in {annot_file}")
                skipped += 1
                continue
            except KeyError as e:
                logger.warning(f"Missing required key in example at line {i}: {e}")
                skipped += 1
                continue

    logger.info(f"Loaded {len(examples)} examples from {annot_file}")
    if skipped > 0:
        logger.warning(f"Skipped {skipped} examples due to errors or missing files")

    return examples


def format_qa_prompt(example: Dict[str, Any]) -> str:
    """
    Create a prompt for the Q->A task following SEED-Bench style.

    Args:
        example: VCR example with tokenized question and answer choices

    Returns:
        Formatted prompt for the Q->A task
    """
    objects = example["objects"]
    question_text = detokenize_text(example["question"], objects)

    # Convert tokenized answer choices to natural language
    answer_choices = [
        detokenize_text(choice, objects) for choice in example["answer_choices"]
    ]

    prompt = "Answer the given multiple-choice question about this image.\n\n"
    prompt += f"Question: {question_text}\n\n"

    for i, choice in enumerate(answer_choices):
        prompt += f"Option {i+1}: {choice}\n"

    prompt += "\nInstructions: Analyze the image and select the most appropriate answer option."
    prompt += "\nProvide your answer as ONLY the option number (1, 2, 3, or 4) with no explanation."
    prompt += "\nYou must choose exactly one option number."
    return prompt


def format_qar_prompt(example: Dict[str, Any]) -> str:
    """
    Create a prompt for the QA->R task using the gold answer.

    Args:
        example: VCR example with tokenized question, answer, and rationale choices

    Returns:
        Formatted prompt for the QA->R task
    """
    objects = example["objects"]
    question_text = detokenize_text(example["question"], objects)

    # Get the correct answer
    correct_answer_idx = example["answer_label"]
    correct_answer = detokenize_text(
        example["answer_choices"][correct_answer_idx], objects
    )

    # Convert tokenized rationale choices to natural language
    rationale_choices = [
        detokenize_text(choice, objects) for choice in example["rationale_choices"]
    ]

    prompt = "Select the best rationale that explains the correct answer to a question about this image.\n\n"
    prompt += f"Question: {question_text}\n"
    prompt += f"Correct answer: {correct_answer}\n\n"
    prompt += "Select the rationale that best explains why this answer is correct:\n"

    for i, choice in enumerate(rationale_choices):
        prompt += f"Rationale {i+1}: {choice}\n"

    prompt += (
        "\nInstructions: Analyze the image and select the most appropriate rationale."
    )
    prompt += "\nProvide your answer as ONLY the rationale number (1, 2, 3, or 4) with no explanation."
    prompt += "\nYou must choose exactly one option number."

    return prompt


def extract_option_number(response: str) -> Optional[int]:
    """
    Extract the option number from model response, following MM-Eval parsing approach.

    Args:
        response: Text response from the model

    Returns:
        0-indexed option number (0-3) or None if no option could be extracted
    """
    if not response:
        return None

    # First try to find a standalone digit
    matches = re.findall(r"\b[1-4]\b", response)
    if matches:
        return int(matches[0]) - 1  # Convert to 0-indexed

    # Next, look for option/answer/rationale followed by number
    matches = re.findall(r"(?:option|answer|rationale)[^\d]*([1-4])", response.lower())
    if matches:
        return int(matches[0]) - 1

    # If still not found, look for any digit
    matches = re.findall(r"[1-4]", response)
    if matches:
        return int(matches[0]) - 1

    return None  # Could not extract an option number


def run_vcr_task(
    model_manager,
    data_dir: str,
    subset_size: Optional[int] = None,
    results_dir: str = "results",
    split: str = "val",
):
    """
    Run Visual Commonsense Reasoning task for all models on the VCR dataset.

    Args:
        model_manager: Manager object for querying different models
        data_dir: Path to the VCR dataset directory
        subset_size: Maximum number of examples to evaluate (None for all)
        results_dir: Directory to save results and metrics
        split: Dataset split to use ("train", "val", or "test")

    Returns:
        Dictionary with evaluation metrics for each model
    """
    logger.info("Starting Visual Commonsense Reasoning task")

    # Load VCR dataset from the specified local directory
    dataset = load_vcr_dataset(data_dir, split=split, sample_size=subset_size)

    if not dataset:
        logger.error("No valid examples loaded from the dataset. Aborting task.")
        return {"error": "No valid examples in dataset"}

    # Initialize results structure for each model
    results = {"openai": [], "anthropic": [], "gemini": []}

    # Create a timestamp-based directory for this run
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"vcr_{split}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save intermediate results after every N examples
    save_interval = 50
    last_save_idx = 0

    # Process each example in the dataset
    for i, example in enumerate(dataset):
        example_id = example.get("annot_id", str(i))
        image_path = example["image_path"]

        logger.info(f"Processing example {i+1}/{len(dataset)} (ID: {example_id})")
        # logger.info(f"Image: {image_path}")

        try:
            # Task 1: Q->A (answer selection)
            qa_prompt = format_qa_prompt(example)
            # logger.info(f"Q->A Prompt:\n{qa_prompt}")

            # Use the existing model_manager to query all models
            qa_responses = model_manager.query_all_models(qa_prompt, image_path)

            # Parse selections from responses
            qa_selections = {}
            for model, response in qa_responses.items():
                selection = extract_option_number(response)
                qa_selections[model] = selection

            # Task 2: QA->R (rationale selection)
            qar_prompt = format_qar_prompt(example)
            # logger.info(f"QA->R Prompt:\n{qar_prompt}")

            # Use the existing model_manager to query all models
            qar_responses = model_manager.query_all_models(qar_prompt, image_path)

            # Parse selections from responses
            qar_selections = {}
            for model, response in qar_responses.items():
                selection = extract_option_number(response)
                qar_selections[model] = selection
                # logger.info(f"{model} QA->R selection: {selection}")

            # Store responses and selections in results
            for model in results:
                results[model].append(
                    {
                        "example_id": example_id,
                        "image_path": str(image_path),
                        "qa_prompt": qa_prompt,
                        "qa_response": qa_responses.get(model, "Error: No response"),
                        "qa_selection": qa_selections.get(model),
                        "qa_correct_answer": example["answer_label"],
                        "qar_prompt": qar_prompt,
                        "qar_response": qar_responses.get(model, "Error: No response"),
                        "qar_selection": qar_selections.get(model),
                        "qar_correct_rationale": example["rationale_label"],
                    }
                )

        except Exception as e:
            logger.error(f"Error processing example {example_id}: {str(e)}")
            error_message = f"Error: {str(e)}"

            # Add error entry for all models
            for model in results:
                results[model].append(
                    {
                        "example_id": example_id,
                        "image_path": (
                            str(image_path) if "image_path" in locals() else "Unknown"
                        ),
                        "qa_prompt": (
                            qa_prompt
                            if "qa_prompt" in locals()
                            else "Error: prompt not created"
                        ),
                        "qa_response": error_message,
                        "qa_selection": None,
                        "qa_correct_answer": (
                            example["answer_label"]
                            if "answer_label" in example
                            else None
                        ),
                        "qar_prompt": (
                            qar_prompt
                            if "qar_prompt" in locals()
                            else "Error: prompt not created"
                        ),
                        "qar_response": error_message,
                        "qar_selection": None,
                        "qar_correct_rationale": (
                            example["rationale_label"]
                            if "rationale_label" in example
                            else None
                        ),
                    }
                )

        # Save intermediate results at regular intervals
        if (i + 1) % save_interval == 0 or i == len(dataset) - 1:
            intermediate_metrics = calculate_metrics(results)
            save_results(
                results, intermediate_metrics, run_dir, is_final=(i == len(dataset) - 1)
            )
            logger.info(
                f"Saved intermediate results after {i+1}/{len(dataset)} examples"
            )
            last_save_idx = i + 1

    # Calculate final metrics and save results
    metrics = calculate_metrics(results)
    save_results(results, metrics, run_dir, is_final=True)

    return metrics


def calculate_metrics(results):
    """
    Calculate comprehensive VCR metrics following the original evaluation approach.

    Args:
        results: Dictionary mapping model names to lists of example results

    Returns:
        Dictionary with metrics for each model
    """
    metrics = {}

    for model, model_results in results.items():
        # Extract predictions and labels for all processed examples
        processed_count = len(model_results)

        # Initialize arrays to track valid examples
        valid_examples = []
        answer_preds = []
        rationale_preds = []
        answer_labels = []
        rationale_labels = []

        for i, result in enumerate(model_results):
            # Skip only if ground truth is missing
            if (
                result["qa_correct_answer"] is None
                or result["qar_correct_rationale"] is None
            ):
                continue

            valid_examples.append(i)
            # For invalid responses, use a value that won't match any correct answer (e.g., -1)
            answer_preds.append(
                result["qa_selection"] if result["qa_selection"] is not None else -1
            )
            rationale_preds.append(
                result["qar_selection"] if result["qar_selection"] is not None else -1
            )
            answer_labels.append(result["qa_correct_answer"])
            rationale_labels.append(result["qar_correct_rationale"])

        # Convert to numpy arrays
        if not valid_examples:  # Handle case with no valid examples
            model_metrics = {
                "q->a_accuracy": 0.0,
                "qa->r_accuracy": 0.0,
                "q->ar_accuracy": 0.0,
                "q->a_count": 0,
                "qa->r_count": 0,
                "q->ar_count": 0,
                "total": 0,
                "processed": processed_count,
            }
            metrics[model] = model_metrics
            continue

        answer_preds = np.array(answer_preds)
        rationale_preds = np.array(rationale_preds)
        answer_labels = np.array(answer_labels)
        rationale_labels = np.array(rationale_labels)

        # Calculate hits (correct predictions)
        answer_hits = answer_preds == answer_labels
        rationale_hits = rationale_preds == rationale_labels
        joint_hits = answer_hits & rationale_hits

        # Calculate metrics
        valid_count = len(valid_examples)  # Number of examples with valid ground truth

        # Use valid_count for counts, but processed_count for accuracy calculations
        model_metrics = {
            "q->a_accuracy": (
                float(np.sum(answer_hits) / processed_count)
                if processed_count > 0
                else 0.0
            ),
            "qa->r_accuracy": (
                float(np.sum(rationale_hits) / processed_count)
                if processed_count > 0
                else 0.0
            ),
            "q->ar_accuracy": (
                float(np.sum(joint_hits) / processed_count)
                if processed_count > 0
                else 0.0
            ),
            # Add counts for reference
            "q->a_count": int(np.sum(answer_hits)),
            "qa->r_count": int(np.sum(rationale_hits)),
            "q->ar_count": int(np.sum(joint_hits)),
            "total": valid_count,
            "processed": processed_count,
        }

        metrics[model] = model_metrics

    return metrics


def save_results(results, metrics, results_dir, is_final=False):
    """
    Save detailed results and metrics to files.

    Args:
        results: Dictionary mapping model names to lists of example results
        metrics: Dictionary with evaluation metrics for each model
        results_dir: Directory to save results
        is_final: Whether this is the final save (as opposed to intermediate)
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    # Save full results for each model
    for model, model_results in results.items():
        suffix = "final" if is_final else "intermediate"
        output_file = results_path / f"vcr_results_{model}_{suffix}.json"
        with open(output_file, "w") as f:
            json.dump(model_results, f, indent=2)

        if is_final:
            logger.info(f"Saved final {model} results to {output_file}")

    # Save metrics summary
    suffix = "final" if is_final else "intermediate"
    metrics_file = results_path / f"vcr_metrics_{suffix}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if is_final:
        logger.info(f"Saved final metrics to {metrics_file}")

    # Create a human-readable summary report
    report_file = results_path / f"vcr_summary_report_{suffix}.md"
    with open(report_file, "w") as f:
        f.write("# Visual Commonsense Reasoning Results\n\n")

        # Add timestamp
        from datetime import datetime

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Model | Q→A | QA→R | Q→AR (Joint) |\n")
        f.write("|-------|-----|------|-------------|\n")

        for model, model_metrics in metrics.items():
            qa = model_metrics["q->a_accuracy"] * 100
            qar = model_metrics["qa->r_accuracy"] * 100
            joint = model_metrics["q->ar_accuracy"] * 100
            f.write(f"| {model} | {qa:.2f}% | {qar:.2f}% | {joint:.2f}% |\n")

        f.write("\n## Detailed Statistics\n\n")
        for model, model_metrics in metrics.items():
            f.write(f"### {model}\n")
            f.write(f"- Total valid examples: {model_metrics['total']}\n")
            f.write(f"- Total processed examples: {model_metrics['processed']}\n")
            f.write(
                f"- Correct answers: {model_metrics['q->a_count']} ({model_metrics['q->a_accuracy']*100:.2f}%)\n"
            )
            f.write(
                f"- Correct rationales: {model_metrics['qa->r_count']} ({model_metrics['qa->r_accuracy']*100:.2f}%)\n"
            )
            f.write(
                f"- Correct answer+rationale pairs: {model_metrics['q->ar_count']} ({model_metrics['q->ar_accuracy']*100:.2f}%)\n\n"
            )

    if is_final:
        logger.info(f"Saved final summary report to {report_file}")
