# Multimodal LLM Evaluation

This repository contains code for evaluating commercially available Multimodal Large Language Models (MLLMs) on various vision-language tasks including Visual Question Answering (VQA), Image Captioning (IC), Visual Commonsense Reasoning (VCR), and Visual Dialog (VisDial).

## Setup

1. Clone this repository
2. Configure your API keys in `api_config.json` for the commercial MLLMs you wish to evaluate (e.g., GPT-4o, Claude 3.7 Sonnet, Gemini 2.0 Pro)

## Running Evaluations

You can run evaluations for different tasks using the following command with arguments (Example for VQA):

python main.py --task vqa --subset-size 50 --results-dir results/vqa

With tasks being defined as "vqa", "ic", "vcr", "visdial".

## Important Notes

- **API Costs**: Be cautious when setting large `subset-size` values as this will increase the number of API calls to commercial services and may result in significant costs.

- **Dataset Requirements**: For VQA, VCR, and VisDia tasks, you need to download the respective datasets and ensure the paths in the configuration are correct before running evaluations.

- **Commercial Focus**: This is designed specifically for evaluating commercial MLLMs via their APIs, not open-source or locally run models.

- **Virtual Environment**: It is recommended to run this code in a virtual environment to avoid conflicts with other Python packages.

## Results

Results will be saved in the specified `results-dir` for further analysis.
