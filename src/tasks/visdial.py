import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
from typing import Optional

logger = logging.getLogger("visual_dialog_task")


def load_visual_dialog_dataset(
    data_dir: str, split: str = "val", sample_size: Optional[int] = None
):
    """
    Load Visual Dialog dataset from local files.

    Args:
        data_dir: Path to the Visual Dialog dataset directory
        split: Dataset split ("train", "val", or "test")
        sample_size: Maximum number of examples to load (None for all)

    Returns:
        List of examples with image paths and processed data
    """
    logger.info(
        f"Loading Visual Dialog dataset from {data_dir}, {split} split"
        + (f" (limited to {sample_size} examples)" if sample_size else "")
    )

    # Construct path to the annotations file
    annot_file = os.path.join(data_dir, f"visdial_1.0_{split}.json")

    if not os.path.exists(annot_file):
        raise FileNotFoundError(f"VisDialog annotations file not found at {annot_file}")

    # Load the annotation file
    with open(annot_file, "r") as f:
        data = json.load(f)

    # Get global lists
    questions = data["data"]["questions"]
    answers = data["data"]["answers"]

    examples = []
    skipped = 0

    # Image directory based on your structure
    image_dir = os.path.join(data_dir, f"VisualDialog_{split}2018")

    # Process dialogs
    for i, dialog in enumerate(data["data"]["dialogs"]):
        if sample_size is not None and len(examples) >= sample_size:
            break

        try:
            # Construct image path
            img_id = dialog["image_id"]

            # Format: VisualDialog_val2018_000000000123
            img_path = os.path.join(
                image_dir, f"VisualDialog_{split}2018_{img_id:012d}.jpg"
            )

            if not os.path.exists(img_path):
                logger.warning(
                    f"Image file not found at {img_path} for image_id {img_id}"
                )
                skipped += 1
                continue

            # Process dialog turns
            processed_dialog = []
            for turn in dialog["dialog"]:
                processed_turn = {
                    "question": questions[turn["question"]],
                    "answer": answers[turn["answer"]],
                    "answer_options": [answers[idx] for idx in turn["answer_options"]],
                    "gt_index": turn["gt_index"],
                }
                processed_dialog.append(processed_turn)

            # Create the example
            example = {
                "image_id": img_id,
                "image_path": img_path,
                "caption": dialog["caption"],
                "dialog": processed_dialog,
            }

            examples.append(example)

        except Exception as e:
            logger.warning(f"Error processing dialog {i}: {str(e)}")
            skipped += 1

    logger.info(f"Loaded {len(examples)} examples from {annot_file}")
    if skipped > 0:
        logger.warning(f"Skipped {skipped} examples due to errors or missing files")

    return examples


def format_ranking_prompt(
    caption: str,
    history: List[Dict[str, str]],
    question: str,
    answer_options: List[str],
) -> str:
    """
    Create a prompt for ranking answer options.

    Args:
        caption: Image caption
        history: List of previous QA pairs
        question: Current question to answer
        answer_options: List of answer options to rank

    Returns:
        Formatted prompt for ranking
    """
    # Format the prompt with image caption
    prompt = f"Given the Image with the caption: {caption}\n\n"

    # Add conversation history
    if history:
        prompt += "The conversation:\n"
        for i, qa_pair in enumerate(history):
            prompt += f"Q{i+1}: {qa_pair['question']}\n"
            prompt += f"A{i+1}: {qa_pair['answer']}\n"
        prompt += "\n"

    # Add current question
    prompt += f"The current question: {question}\n\n"

    # Add answer options
    prompt += "Rank the following answer options from best to worst.\n"
    for i, option in enumerate(answer_options):
        prompt += f"{i+1}. {option}\n"

    # Add response format instructions
    prompt += "\nRespond only with a comma-separated list of the answer option numbers, ordered from best to worst (e.g., '5,2,9,1,3,...')."

    return prompt


def parse_ranking_response(response: str, num_options: int) -> List[int]:
    """
    Parse a ranking response from the model into a list of indices.

    Args:
        response: Text response from the model
        num_options: Number of options to rank

    Returns:
        List of 0-indexed option indices in ranked order
    """
    # Extract numbers from the response
    numbers = re.findall(r"\d+", response)

    # Convert to integers and make 0-indexed
    ranking = []
    seen = set()

    for num in numbers:
        try:
            idx = int(num) - 1  # Convert to 0-indexed

            # Only add valid and non-duplicate indices
            if 0 <= idx < num_options and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
        except ValueError:
            continue

    # Add any missing indices at the end
    for i in range(num_options):
        if i not in seen:
            ranking.append(i)

    # Ensure we have exactly num_options items
    return ranking[:num_options]


def calculate_metrics(
    rankings: List[List[int]], gt_indices: List[int]
) -> Dict[str, float]:
    """
    Calculate Visual Dialog retrieval metrics.

    Args:
        rankings: List of model rankings (each ranking is a list of indices)
        gt_indices: List of ground truth answer indices

    Returns:
        Dictionary with metrics (MRR, R@k, mean rank)
    """
    if not rankings or not gt_indices:
        logger.warning("Empty rankings or ground truth indices")
        return {
            "mrr": 0.0,
            "r@1": 0.0,
            "r@5": 0.0,
            "r@10": 0.0,
            "mean_rank": 0.0,
            "count": 0,
        }

    # Initialize arrays for metrics
    ranks = []

    # Calculate rank of ground truth for each example
    for ranking, gt_idx in zip(rankings, gt_indices):
        try:
            # Find position of ground truth in the ranking (convert to 1-indexed)
            rank = ranking.index(gt_idx) + 1
        except ValueError:
            # If gt_idx not in ranking, assign worst possible rank
            rank = len(ranking)

        ranks.append(rank)

    # Convert to numpy array for easier computation
    ranks = np.array(ranks)

    # Calculate metrics
    mrr = np.mean(1.0 / ranks) if len(ranks) > 0 else 0.0
    r1 = np.mean(ranks <= 1) if len(ranks) > 0 else 0.0
    r5 = np.mean(ranks <= 5) if len(ranks) > 0 else 0.0
    r10 = np.mean(ranks <= 10) if len(ranks) > 0 else 0.0
    mean_rank = np.mean(ranks) if len(ranks) > 0 else 0.0

    return {
        "mrr": float(mrr),
        "r@1": float(r1),
        "r@5": float(r5),
        "r@10": float(r10),
        "mean_rank": float(mean_rank),
        "count": len(ranks),
    }


def save_results(results, metrics, results_dir, is_final=False):
    """
    Save detailed results and metrics to files.

    Args:
        results: Dictionary mapping model names to lists of example results
        metrics: Dictionary with evaluation metrics for each model
        results_dir: Directory to save results
        is_final: Whether this is the final save (vs intermediate)
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    # Save full results for each model
    for model, model_results in results.items():
        suffix = "final" if is_final else "intermediate"
        output_file = results_path / f"visdial_results_{model}_{suffix}.json"
        with open(output_file, "w") as f:
            json.dump(model_results, f, indent=2)

        if is_final:
            logger.info(f"Saved final {model} results to {output_file}")

    # Save metrics summary
    suffix = "final" if is_final else "intermediate"
    metrics_file = results_path / f"visdial_metrics_{suffix}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if is_final:
        logger.info(f"Saved final metrics to {metrics_file}")

    # Create a human-readable summary report
    report_file = results_path / f"visdial_summary_report_{suffix}.md"
    with open(report_file, "w") as f:
        f.write("# Visual Dialog Results\n\n")

        # Add timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Model | MRR | R@1 | R@5 | R@10 | Mean Rank |\n")
        f.write("|-------|-----|-----|-----|------|----------|\n")

        for model, model_metrics in metrics.items():
            mrr = model_metrics["mrr"] * 100
            r1 = model_metrics["r@1"] * 100
            r5 = model_metrics["r@5"] * 100
            r10 = model_metrics["r@10"] * 100
            mean_rank = model_metrics["mean_rank"]
            f.write(
                f"| {model} | {mrr:.2f}% | {r1:.2f}% | {r5:.2f}% | {r10:.2f}% | {mean_rank:.2f} |\n"
            )

        f.write("\n## Detailed Statistics\n\n")
        for model, model_metrics in metrics.items():
            f.write(f"### {model}\n")
            f.write(f"- Total examples: {model_metrics['count']}\n")
            f.write(f"- MRR: {model_metrics['mrr']:.4f}\n")
            f.write(f"- R@1: {model_metrics['r@1']:.4f}\n")
            f.write(f"- R@5: {model_metrics['r@5']:.4f}\n")
            f.write(f"- R@10: {model_metrics['r@10']:.4f}\n")
            f.write(f"- Mean Rank: {model_metrics['mean_rank']:.4f}\n\n")

    if is_final:
        logger.info(f"Saved final summary report to {report_file}")


def run_visdial_task(
    model_manager,
    data_dir: str = "src/tasks/data/visdial",
    subset_size: Optional[int] = None,
    results_dir: str = "results",
    split: str = "val",
    last_round_only: bool = True,
):
    """
    Run Visual Dialog task for all models.

    Args:
        model_manager: Manager object for querying different models
        data_dir: Path to the Visual Dialog dataset directory
        subset_size: Maximum number of examples to evaluate
        results_dir: Directory to save results and metrics
        split: Dataset split to use
        last_round_only: Whether to evaluate only the last round of each dialog

    Returns:
        Dictionary with evaluation metrics for each model
    """
    logger.info(f"Starting Visual Dialog task (last_round_only={last_round_only})")

    # Load dataset
    dataset = load_visual_dialog_dataset(data_dir, split=split, sample_size=subset_size)

    if not dataset:
        logger.error("No valid examples loaded from the dataset. Aborting task.")
        return {"error": "No valid examples in dataset"}

    # Initialize results structure for each model
    results = {"openai": [], "anthropic": [], "gemini": []}

    # Create timestamp-based directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"visdial_{split}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save interval
    save_interval = 10

    # Store rankings and ground truths for metric calculation
    all_rankings = {"openai": [], "anthropic": [], "gemini": []}
    all_gt_indices = []

    # Process each example
    for i, example in enumerate(dataset):
        example_id = example.get("image_id", str(i))
        image_path = example["image_path"]

        logger.info(f"Processing example {i+1}/{len(dataset)} (ID: {example_id})")

        try:
            if last_round_only:
                # Only process the last turn of each dialog
                turn_indices = [len(example["dialog"]) - 1]
            else:
                # Process all turns
                turn_indices = range(len(example["dialog"]))

            for turn_idx in turn_indices:
                # Get current turn data
                turn = example["dialog"][turn_idx]
                question = turn["question"]
                answer_options = turn["answer_options"]
                gt_index = turn["gt_index"]

                # Get conversation history (all turns before current)
                history = []
                for prev_idx in range(turn_idx):
                    history.append(
                        {
                            "question": example["dialog"][prev_idx]["question"],
                            "answer": example["dialog"][prev_idx]["answer"],
                        }
                    )

                # Format prompt
                prompt = format_ranking_prompt(
                    caption=example["caption"],
                    history=history,
                    question=question,
                    answer_options=answer_options,
                )

                # Query all models
                model_responses = model_manager.query_all_models(prompt, image_path)

                # Parse rankings from responses
                rankings = {}
                for model, response in model_responses.items():
                    try:
                        ranking = parse_ranking_response(response, len(answer_options))
                        rankings[model] = ranking

                        # Store ranking for metrics calculation
                        all_rankings[model].append(ranking)
                    except Exception as e:
                        logger.error(f"Error parsing ranking for {model}: {e}")
                        # Use random ranking as fallback
                        fallback_ranking = list(range(len(answer_options)))
                        np.random.shuffle(fallback_ranking)
                        rankings[model] = fallback_ranking
                        all_rankings[model].append(fallback_ranking)

                # Store ground truth
                all_gt_indices.append(gt_index)

                # Store results for each model
                for model in results:
                    results[model].append(
                        {
                            "example_id": example_id,
                            "turn_idx": turn_idx,
                            "image_path": str(image_path),
                            "prompt": prompt,
                            "response": model_responses.get(model, ""),
                            "ranking": rankings.get(model, []),
                            "gt_index": gt_index,
                            "options": answer_options,
                        }
                    )

        except Exception as e:
            logger.error(f"Error processing example {example_id}: {str(e)}")
            # Continue with next example

        # Save intermediate results
        if (i + 1) % save_interval == 0 or i == len(dataset) - 1:
            # Calculate intermediate metrics
            intermediate_metrics = {}
            for model in results:
                model_rankings = all_rankings[model]
                intermediate_metrics[model] = calculate_metrics(
                    model_rankings, all_gt_indices
                )

            # Save intermediate results
            save_results(
                results, intermediate_metrics, run_dir, is_final=(i == len(dataset) - 1)
            )
            logger.info(
                f"Saved intermediate results after {i+1}/{len(dataset)} examples"
            )

    # Calculate final metrics
    final_metrics = {}
    for model in results:
        model_rankings = all_rankings[model]
        final_metrics[model] = calculate_metrics(model_rankings, all_gt_indices)

    # Save final results
    save_results(results, final_metrics, run_dir, is_final=True)

    return final_metrics
