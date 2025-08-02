# Technical Specification: rewrite_function.py

**Date:** 2025-04-26

## 1. File Overview

* **Purpose:** This module provides core functionality for rewriting text based on specific improvement suggestions. It primarily uses a locally loaded, quantized Gemma 27B model. Key components include loading the model (with caching), defining a custom client for Autogen compatibility (`CustomGemmaClient`), and the main `rewrite_text` function that orchestrates the prompting and generation process. It's designed to be imported and used by other scripts in the pipeline (specifically `step3.py`).
* **Role in Project:** Serves as the central text improvement engine, taking original text and feedback points to generate a revised version using the LLM. It encapsulates the LLM loading, interaction, and prompt engineering needed for the rewriting task.

## 2. Key Classes

* **`CustomGemmaClient`**
  * **Responsibility:** Provides an interface compatible with the Autogen framework for interacting with the locally loaded Gemma LLM. It handles Gemma-specific prompt formatting (`_format_chat_prompt`) and manages the call to the underlying Hugging Face model for generation.
  * **Key Attributes:** `config` (model/client config passed from Autogen), `model_name`, `device`, `gen_params` (generation parameters), `model`, `tokenizer`.
  * **Key Methods:**
    * `__init__`: Initializes the client, sets up model name, device, and generation parameters. Crucially, calls `load_shared_model` to get the cached model instance.
    * `_format_chat_prompt`: Formats a list of Autogen messages into the specific chat template required by Gemma (`<start_of_turn>role\ncontent<end_of_turn>`).
    * `create`: The main method called by Autogen. Takes message parameters, formats the prompt, calls `model.generate()` with configured parameters (including error handling and a fallback to greedy decoding), cleans the response format, and returns it in an Autogen-compatible structure (`SimpleNamespace`).
    * `message_retrieval`: Extracts the generated text content from the response structure.
    * `cost`, `get_usage`: Placeholder methods for cost/token tracking (return 0/empty dict).
  * **Core Functionality:** Acts as a bridge between the Autogen framework (used in `step3.py`) and the underlying `transformers` model (`load_shared_model`), managing prompt formatting and the generation call.

## 3. Key Functions

* **`monitor_gpu_memory()`**: Utility function to log current GPU memory usage statistics (total, reserved, allocated, free) for debugging.
* **`load_shared_model(model_name, device)`**: Loads the specified Hugging Face model (Gemma 27B by default) and tokenizer using `transformers`.
  * Implements in-memory caching (`_model_cache`) to avoid reloading the large model repeatedly across different calls or modules.
  * Configures 4-bit quantization (`BitsAndBytesConfig`), GPU memory allocation (`max_memory`), and device mapping (`device_map="auto"` with offloading).
  * Includes robust error handling and memory management (clearing cache).
  * Handles tokenizer setup (setting `pad_token`).
  * Returns the loaded `model` and `tokenizer`.
* **`clean_response(text)`**: Post-processes the raw text output from the LLM. Removes Gemma-specific chat template artifacts (`<start_of_turn>`, `<end_of_turn>`, `model\n`) and attempts to extract only the core rewritten content. It also handles cases where JSON formatting instructions might leak into the response.
* **`rewrite_text(improvement_points, original_text, temperature=0.7, max_tokens=2000, referenced_papers=None, full_chapter_context=None)`**: The primary function exported for use by other modules.
  * Takes improvement suggestions, the original text, and optional context (references, full chapter) as input.
  * Calls `create_rewrite_prompt` to construct the detailed prompt for the LLM.
  * Calls `call_model` to execute the generation request using the `CustomGemmaClient`.
  * Calls `clean_response` to purify the LLM's output.
  * Includes retry logic (up to `max_retries=2`) in case the initial rewrite attempt fails or produces empty output.
  * Returns the cleaned, rewritten text.
* **`call_model(prompt, system_message=None, temperature=0.7, max_tokens=1000)`**: A wrapper function to interact with the `CustomGemmaClient` in a non-Autogen context (used directly by `rewrite_text`).
  * Constructs the message list (user prompt, optional system message).
  * Instantiates `CustomGemmaClient` (which loads the shared model).
  * Calls the client's `create` method.
  * Retrieves the response using `message_retrieval`.
  * Returns the raw response content.
* **`extract_from_json(json_file_path)`**: Reads a JSON file (expected to be the output format from `step3.py`'s `save_consolidated_output`), extracts relevant fields like `original_text`, `improvements` (parsed from moderator feedback), `referenced_papers`, and `full_chapter_context`. Used primarily for standalone testing/debugging of the rewrite process.
* **`create_rewrite_prompt(original_text, improvement_points, referenced_papers=None, full_chapter_context=None)`**: Constructs the detailed prompt used by `rewrite_text`. It includes instructions, the `original_text`, the `improvement_points`, and optionally context from `referenced_papers` and the `full_chapter_context`. The prompt guides the model to revise the text based *only* on the provided feedback.
* **`estimate_memory_needs(text_length)`**: A simple heuristic function to estimate potential GPU memory needs based on input text length (appears unused in the main rewrite flow).
* **`batch_process_text(original_text, improvement_points, max_length=5000, referenced_papers=None, full_chapter_context=None)`**: Handles rewriting potentially long texts by splitting them into batches (based on `max_length`), rewriting each batch individually using `rewrite_text`, and then concatenating the results. Includes logic for maintaining context across batches (though implementation details are complex).
* **`rewrite_text_from_json(json_file_path, temperature=0.7, max_tokens=2000)`**: Combines `extract_from_json` and `rewrite_text` (or potentially `batch_process_text`) to perform a rewrite directly from a saved JSON output file (likely for testing).
* **`process_folder(folder_path='outputs', temperature=0.7, max_tokens=2000)`**: Iterates through JSON files in a specified folder (presumably containing outputs from `step3`), calls `rewrite_text_from_json` for each, and saves the rewritten text back to a new file or updates the existing one. (Primarily for batch testing/offline processing).
* **`main()`**: Entry point for standalone execution.
  * Parses command-line arguments (input JSON file path, temperature, max tokens).
  * Calls `rewrite_text_from_json` with the provided arguments.
  * Prints the rewritten text to the console.

## 4. Data Structures / Constants

* **Key Data Structure:** `_model_cache` (Dict): Global dictionary used by `load_shared_model` for caching the model and tokenizer.
* **Constants:** Default model name (`google/gemma-3-27b-it`), default generation parameters (`temperature`, `max_tokens`), prompt templates used in `create_rewrite_prompt`.
* **Input/Output (for standalone use):** Reads/writes JSON files structured similarly to the outputs of `step3.py` when run directly via `main` or `process_folder`.

## 5. Logic Flow & Control

* **Primary Use Case (as imported module):**
    1. `step3.py` calls `load_shared_model` (likely once at its start).
    2. `step3.py` calls `rewrite_text` with original text and improvements.
    3. `rewrite_text` calls `create_rewrite_prompt`.
    4. `rewrite_text` calls `call_model`.
    5. `call_model` instantiates `CustomGemmaClient` (using the cached model).
    6. `call_model` calls `client.create`, which formats the prompt and triggers `model.generate()`.
    7. The result flows back up, is cleaned by `clean_response` in `rewrite_text`, and returned to `step3.py`.
* **Standalone Execution (`if __name__ == "__main__":`)**
    1. Parses command-line arguments (input JSON file).
    2. Calls `rewrite_text_from_json`.
    3. `rewrite_text_from_json` calls `extract_from_json` to load data.
    4. `rewrite_text_from_json` calls `rewrite_text` (or potentially `batch_process_text`).
    5. The rewrite process follows the steps above (prompt creation, model call, cleaning).
    6. The final rewritten text is printed.
* **Control Flow:** Includes retry logic within `rewrite_text`. The `batch_process_text` function implements looping for handling large inputs. Error handling (try-except) is present in model loading and generation calls.

## 6. External Interactions

* **Imports:** `time`, `sys`, `json`, `os`, `logging`, `torch`, `types`, `transformers`, `dotenv`, `huggingface_hub`, `argparse`, `re`.
* **File System Reads:**
  * `.env` file (via `dotenv`).
  * JSON files when run standalone (`extract_from_json`, `process_folder`).
* **File System Writes:**
  * Log messages (to console by default).
  * Potentially writes model layers to `./offload/`.
  * Rewritten text files when using `process_folder`.
* **External Libraries Called:**
  * `transformers`: Core library for models (`AutoModelForCausalLM`), tokenizers (`AutoTokenizer`), quantization (`BitsAndBytesConfig`), generation (`model.generate`).
  * `torch`: Tensor operations, GPU management (`cuda.empty_cache`, device mapping).
  * `huggingface_hub`: Authentication (`login`).
  * `dotenv`: Loading `.env`.
  * `json`: Loading data for standalone use.
  * `re`: Used within `clean_response`.
  * `logging`: Recording execution details.
* **Local Modules Called:** None. This module provides functions *to* other modules.
* **Exports/Intended Use by Others:**
  * `rewrite_text`: The main function intended for use by `step3.py`.
  * `load_shared_model`: Used by `step3.py` and potentially `step2.py` and `step4.py` to ensure the same model instance is loaded.
  * `CustomGemmaClient`: The Autogen-compatible client class, needed by `step3.py` (and potentially `step2.py`) when setting up the Autogen environment.

## 7. Assumptions / Dependencies

* **Environment:** Assumes a Python environment with necessary libraries installed (`transformers`, `torch` (with CUDA), `huggingface_hub`, `dotenv`, `accelerate`, `bitsandbytes`).
* **Hardware:** Critically depends on a compatible NVIDIA GPU with sufficient VRAM (~70GiB configured) for the Gemma 27B model.
* **API Keys/Tokens:** Requires `HUGGINGFACE_TOKEN` to be set in the environment or `.env` file for model access.
* **Input Format:** When used via `rewrite_text`, assumes `improvement_points` is a list of strings and `original_text` is a string. When run standalone, assumes input JSON files match the expected structure derived from `step3.py`'s output.
* **File Permissions:** Requires write access to `./offload/` if offloading occurs. Read access for `.env`. Read/write access to input/output folders when using `process_folder`.
* **Model Behavior:** Assumes the LLM can understand the rewrite prompt and generate improved text based on the feedback, staying within the requested format. Relies on `clean_response` to handle model output variations.
