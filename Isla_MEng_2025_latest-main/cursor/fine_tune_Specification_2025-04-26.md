# Technical Specification: fine_tune.py

**Date:** 2025-04-26

## 1. File Overview

* **Purpose:** This script performs fine-tuning on a pre-trained transformer model (specifically Mistral-7B-Instruct-v0.2) using custom text data. The goal is likely to adapt the model's writing style or knowledge based on the content of the improved report chapters generated in the previous steps. It uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) and 4-bit quantization (via `bitsandbytes`) to make the process feasible on available hardware.
* **Role in Project:** Optionally adapts a base LLM using the project's generated content (`./chapter_markdowns/` or potentially JSON outputs) to potentially improve its performance or stylistic alignment for future generation tasks within the project or related applications. The output is a fine-tuned model saved locally.

## 2. Key Classes

* No key classes are defined within this script. It primarily uses functions and classes imported from the `transformers`, `datasets`, and `peft` libraries.

## 3. Key Functions

* **`load_text_files(directory)`**: Reads all `.md` files from the specified `directory` (`./chapter_markdowns` by default), extracts their content, and returns a list of strings, where each string is the full content of a Markdown file. Includes error handling for missing directories or files.
* **`prepare_dataset(texts, tokenizer)`**: Takes a list of text strings and a Hugging Face tokenizer. It tokenizes the texts (with truncation and padding to `max_length=512`), and converts them into a Hugging Face `Dataset` object suitable for use with the `Trainer`.
* **`load_json_files(directory)`**: (Note: This function is defined but **not called** in the `main` function of the provided script). Reads `.json` files from a specified directory, presumably containing the detailed output from `step3.py`. It specifically looks for the 'improved' content within sections ('INTRODUCTION', 'METHODOLOGY', etc.) inside the JSON structure and formats them into text strings (e.g., `"### INTRODUCTION:\n{improved_text}\n\n"`). Returns a list of these formatted text strings.
* **`main()`**: The main execution function that orchestrates the fine-tuning process.
  * Sets the base `model_name` (Mistral-7B-Instruct-v0.2).
  * Specifies the input `directory` for training data (`./chapter_markdowns`).
  * Configures 4-bit quantization using `BitsAndBytesConfig`.
  * Loads the pre-trained model (`AutoModelForCausalLM.from_pretrained`) with the quantization config and `device_map="auto"`.
  * Prepares the model for k-bit training using `prepare_model_for_kbit_training` (enables gradient checkpointing, etc.).
  * Configures LoRA using `LoraConfig` (specifying rank, alpha, target modules, dropout).
  * Applies LoRA configuration to the model using `get_peft_model`. Prints trainable parameters.
  * Loads the corresponding tokenizer (`AutoTokenizer.from_pretrained`) and sets the pad token.
  * Calls `load_text_files` to get the training data from Markdown files.
  * Calls `prepare_dataset` to tokenize and format the data.
  * Defines training arguments using `TrainingArguments` (output directory, epochs, batch size, learning rate, save steps, gradient accumulation/checkpointing, fp16, etc.).
  * Creates a `DataCollatorForLanguageModeling` for causal language modeling.
  * Initializes the `Trainer` with the model, arguments, dataset, and data collator.
  * Starts the fine-tuning process by calling `trainer.train()`.
  * Saves the final fine-tuned LoRA adapter weights and tokenizer to the specified output directory (`./fine_tuned_model`).

## 4. Data Structures / Constants

* **Input Data:** Reads text content from `.md` files located in the `directory` specified in `main` (`./chapter_markdowns`). (Alternatively, could read structured JSON from `load_json_files` if that function were used).
* **Output Data:** Saves the fine-tuned model adapters (`adapter_model.safetensors`, `adapter_config.json`, etc.) and tokenizer files to the directory specified in `TrainingArguments` (`./fine_tuned_model`).
* **Constants:** `model_name` ("mistralai/Mistral-7B-Instruct-v0.2"), input `directory`, LoRA configuration parameters (`r`, `lora_alpha`, `target_modules`, etc.), `TrainingArguments` parameters (paths, hyperparameters).

## 5. Logic Flow & Control

* Script execution starts in `main()` when run directly.
* Environment setup: Load `.env`, Hugging Face login.
* Configuration: Define model name, input directory, quantization config, LoRA config.
* Model Loading: Load base model with quantization, prepare for k-bit training, apply LoRA adapters.
* Tokenizer Loading: Load tokenizer, set pad token.
* Data Loading: Call `load_text_files` to read Markdown content.
* Data Preparation: Call `prepare_dataset` to tokenize and create Hugging Face `Dataset`.
* Training Setup: Define `TrainingArguments`, create `DataCollator`.
* Trainer Initialization: Create `Trainer` instance.
* Training Execution: Call `trainer.train()` to start the fine-tuning loop.
* Saving: After training completes, call `model.save_pretrained` and `tokenizer.save_pretrained` to save the results.
* Error handling is present in file loading functions (`load_text_files`, `load_json_files`). The `transformers` library handles errors during model loading and training internally, often raising exceptions.

## 6. External Interactions

* **Imports:** `transformers` (core library for models, tokenizers, training), `datasets` (for handling training data), `peft` (Parameter-Efficient Fine-Tuning library), `torch`, `os`, `json`, `dotenv`, `huggingface_hub`.
* **File System Reads:**
  * `.env` file (via `dotenv`).
  * Markdown files (`.md`) from `./chapter_markdowns/` (via `load_text_files`).
  * (Potentially JSON files from `./outputs/consolidated/` if `load_json_files` were used).
* **File System Writes:**
  * Fine-tuned model adapter files and tokenizer configuration to `./fine_tuned_model/`.
  * Training logs and checkpoints (potentially) to `./fine_tuned_model/` and `./logs/` as configured in `TrainingArguments`.
  * Potentially writes model layers to `./offload/` during loading.
* **External Libraries Called:**
  * `transformers`: `AutoModelForCausalLM`, `AutoTokenizer`, `TrainingArguments`, `Trainer`, `DataCollatorForLanguageModeling`, `BitsAndBytesConfig`.
  * `datasets`: `Dataset`.
  * `peft`: `LoraConfig`, `get_peft_model`, `prepare_model_for_kbit_training`.
  * `torch`: Used for data types (`torch.float16`) and implicitly by `transformers` and `peft` for GPU operations.
  * `huggingface_hub`: Authentication (`login`).
  * `dotenv`: Loading `.env`.
* **Local Modules Called:** None.
* **Exports/Intended Use by Others:** The primary output is the fine-tuned model saved in `./fine_tuned_model`. This adapted model could then potentially be loaded and used in other parts of the project (e.g., modifying `step2.py`, `step3.py`, or `step4.py` to use this fine-tuned version instead of the base model) for improved generation quality or style.

## 7. Assumptions / Dependencies

* **Prior Steps:** Assumes that `step3.py` (or potentially `step2.py`) has run and produced relevant content in the specified input directory (`./chapter_markdowns/`).
* **Environment:** Assumes a Python environment with necessary libraries installed: `transformers`, `datasets`, `peft`, `torch` (with CUDA), `accelerate`, `bitsandbytes`, `huggingface_hub`, `dotenv`.
* **Hardware:** Critically depends on having a compatible NVIDIA GPU with sufficient VRAM to load and fine-tune the Mistral-7B model, even with 4-bit quantization and LoRA. The exact VRAM requirement depends on configuration but is significant.
* **API Keys/Tokens:** Requires `HUGGINGFACE_TOKEN` to be set for Hugging Face Hub login (needed for downloading the base model).
* **Input Data:** Assumes the specified input directory exists and contains `.md` files with sufficient text content for fine-tuning. If `load_json_files` were used, it assumes JSON files with the specific structure exist.
* **File Permissions:** Requires read access to the input directory and write access to the output directory (`./fine_tuned_model`), logging directory (`./logs`), and potentially `./offload/`.
* **Configuration:** Relies on hardcoded model names, directory paths, and hyperparameters within the script. Assumes the chosen hyperparameters are suitable for the task and hardware.
