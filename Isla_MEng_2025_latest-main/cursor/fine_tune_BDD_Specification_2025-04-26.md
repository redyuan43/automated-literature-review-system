# BDD Specification: fine_tune.py

**Date:** 2025-04-26

## Feature: Fine-Tuning a Base LLM with Custom Report Data

As a project developer,
I want to adapt a pre-trained language model (Mistral-7B) using the content of generated report chapters,
So that the model potentially performs better or adopts the style of the custom data for future tasks.

## Key Scenarios

### Scenario: Successful Fine-Tuning Run using Markdown Chapters

* **Given** the input directory `./chapter_markdowns` exists and contains one or more valid Markdown files (`.md`) with sufficient text.
* **And** the output directory `./fine_tuned_model` exists or can be created.
* **And** the base model `mistralai/Mistral-7B-Instruct-v0.2` is accessible via Hugging Face Hub.
* **And** a valid Hugging Face token is provided via environment variables (`HUGGINGFACE_TOKEN`).
* **And** sufficient GPU resources (VRAM, compute) are available to load and train the model with 4-bit quantization and LoRA.
* **When** the `fine_tune.py` script is executed.
* **Then** the script should load the base Mistral-7B model and tokenizer.
* **And** the model should be prepared for training with 4-bit quantization and LoRA adapters applied.
* **And** the text content from all `.md` files in `./chapter_markdowns` should be loaded.
* **And** the loaded texts should be tokenized and formatted into a training dataset.
* **And** the `Trainer` should execute the fine-tuning process for the configured number of epochs.
* **And** training progress (loss, steps) should be logged periodically.
* **And** upon completion, the fine-tuned LoRA adapter weights and the tokenizer should be saved to the `./fine_tuned_model` directory.
* **And** a completion message should be printed.

### Scenario: Input Directory Does Not Exist

* **Given** the specified input directory (e.g., `./chapter_markdowns`) does *not* exist.
* **When** the `fine_tune.py` script is executed and attempts to load data via `load_text_files`.
* **Then** the script should raise a `FileNotFoundError`.
* **And** the fine-tuning process should terminate with an error message indicating the directory was not found.

### Scenario: Input Directory Exists but Contains No Valid Files

* **Given** the input directory `./chapter_markdowns` exists but contains no `.md` files (or only empty ones).
* **When** the `fine_tune.py` script is executed and attempts to load data via `load_text_files`.
* **Then** the `load_text_files` function should return an empty list of texts.
* **And** the script should raise a `ValueError` indicating that no text was extracted.
* **And** the fine-tuning process should terminate with an error message.

### Scenario: Fine-Tuning using Improved JSON Content (Conceptual - Function Not Used in `main`)

* **Given** JSON files exist in a specified directory (e.g., `./outputs/consolidated/`).
* **And** these JSON files contain structured data with an 'improved' text field under specific section keys (e.g., `sections['INTRODUCTION']['improved']`).
* **And** the base model and tokenizer are loaded and prepared for training.
* **When** the `load_json_files` function is called (if it were integrated into `main`).
* **Then** the 'improved' text content from the relevant sections of each JSON file should be extracted.
* **And** this extracted text should be formatted (e.g., prepended with `### SECTION_NAME:`).
* **And** a list of these formatted text strings should be returned.
* **And** this list should be used by `prepare_dataset` to create the training data.
* **And** the fine-tuning process should proceed using this JSON-derived dataset.

## Key Components Involved (Behavioral Roles)

* **`main`:** Orchestrates the entire fine-tuning pipeline: model/tokenizer loading, data loading/preparation, configuring training arguments, initializing and running the trainer, and saving the final model.
* **`load_text_files`:** Responsible for finding and reading the content of Markdown files from the specified input directory to be used as training data.
* **`load_json_files`:** (Defined but unused in `main`) Responsible for finding and extracting specific 'improved' text sections from JSON files to potentially be used as training data.
* **`prepare_dataset`:** Takes raw text data and converts it into a tokenized, formatted Hugging Face `Dataset` ready for the `Trainer`.
* **Transformers `AutoModelForCausalLM` / `AutoTokenizer`:** Used to load the base pre-trained language model and its corresponding tokenizer.
* **Transformers `BitsAndBytesConfig` / PEFT `prepare_model_for_kbit_training`:** Configure and apply 4-bit quantization to the model.
* **PEFT `LoraConfig` / `get_peft_model`:** Configure and apply LoRA adapters to the model for parameter-efficient fine-tuning.
* **Transformers `TrainingArguments`:** Define all hyperparameters and configurations for the training process (epochs, learning rate, batch size, save locations, etc.).
* **Transformers `DataCollatorForLanguageModeling`:** Prepares batches of data correctly for the causal language modeling task during training.
* **Transformers `Trainer`:** Manages the actual training loop, handling optimization, evaluation (if configured), logging, and checkpointing.

## Inputs and Outputs (Behavioral)

* **Input:**
  * Text content from Markdown (`.md`) files in `./chapter_markdowns/`.
  * Base pre-trained model identifier (`mistralai/Mistral-7B-Instruct-v0.2`).
  * Configuration parameters for quantization, LoRA, and training (mostly hardcoded or set in `TrainingArguments`).
  * Hugging Face token from `.env`.
* **Output:**
  * Fine-tuned LoRA adapter weights saved in `./fine_tuned_model/`.
  * Tokenizer files saved in `./fine_tuned_model/`.
  * Training logs and potentially checkpoints saved in `./fine_tuned_model/` and `./logs/`.
  * Console output indicating progress and completion status.

## Interactions with Other Components

* **File System:** Reads `.md` files from `./chapter_markdowns/`; Reads `.env`; Writes model files, tokenizer files, logs, and checkpoints to `./fine_tuned_model/` and `./logs/`; Potentially writes offload files to `./offload/`.
* **`step3.py` Output:** Consumes the Markdown chapter files generated by `step3.py` as its primary training data source.
* **Hugging Face Hub:** Downloads the base model (`mistralai/Mistral-7B-Instruct-v0.2`) and tokenizer using the provided token for authentication.
* **Transformers Library:** Core dependency for model loading, tokenization, quantization, training loop (`Trainer`), and data collation.
* **PEFT Library:** Used to configure and apply LoRA modifications to the base model.
* **Datasets Library:** Used to structure the text data into a format suitable for the `Trainer`.
* **Torch Library:** Underpins the model operations, GPU management, and training process.
* **dotenv Library:** Used to load the Hugging Face token from the `.env` file.
