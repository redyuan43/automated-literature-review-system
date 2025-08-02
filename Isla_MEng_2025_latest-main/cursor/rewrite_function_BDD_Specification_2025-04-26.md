# BDD Specification: rewrite_function.py

**Date:** 2025-04-26

## Feature: LLM-Powered Text Rewriting Service

As a user or another system component (like `step3.py`),
I want to rewrite text based on specific improvement suggestions using an LLM,
So that the quality of the text is iteratively improved according to feedback.

## Key Scenarios

### Scenario: Successfully Rewriting Text Based on Feedback

* **Given** an original text string needs improvement.
* **And** a list of specific improvement points (feedback) is provided.
* **And** the shared LLM (Gemma 27B) is loaded and accessible via `load_shared_model`.
* **When** the `rewrite_text` function is called with the original text and improvement points.
* **Then** a detailed prompt containing the original text and the feedback should be constructed.
* **And** this prompt should be sent to the LLM via the `CustomGemmaClient`.
* **And** the LLM should generate a rewritten version of the text.
* **And** the raw LLM response should be cleaned to remove artifacts (like chat templates).
* **And** the cleaned, rewritten text string should be returned.
* **And** log messages should indicate the start and successful completion of the rewrite process.

### Scenario: Rewriting Text with Additional Context

* **Given** an original text string needs improvement.
* **And** a list of specific improvement points is provided.
* **And** additional context, such as referenced papers and the full chapter text, is available.
* **And** the shared LLM is loaded.
* **When** the `rewrite_text` function is called with the original text, improvements, and the additional context.
* **Then** a prompt should be constructed that includes the original text, feedback, *and* the provided context (references, full chapter).
* **And** the LLM should generate a rewritten version using all provided information.
* **And** the cleaned, rewritten text string should be returned.

### Scenario: Handling Rewrite Failure (e.g., Empty LLM Response)

* **Given** an original text string and improvement points.
* **And** the shared LLM is loaded.
* **And** when the LLM is prompted via `call_model`, it returns an empty or unusable response.
* **When** the `rewrite_text` function attempts to get the rewritten text.
* **Then** the initial call to `call_model` should return the empty/unusable response.
* **And** the `rewrite_text` function should detect the failure.
* **And** it should attempt to retry the call to `call_model` (up to a maximum number of retries).
* **And** if retries also fail, the `rewrite_text` function should return the *original* text string.
* **And** log messages should indicate the failure and the retry attempts.

### Scenario: Loading and Caching the Shared LLM

* **Given** the LLM (`google/gemma-3-27b-it`) has not yet been loaded into the shared cache (`_model_cache`).
* **When** `load_shared_model` is called for the first time (e.g., by `CustomGemmaClient` or another script).
* **Then** the script should log that it is loading the model and tokenizer.
* **And** the model should be loaded from Hugging Face with the specified quantization and memory configurations.
* **And** the tokenizer should be loaded.
* **And** the loaded model and tokenizer should be stored in the `_model_cache`.
* **And** the loaded model and tokenizer should be returned.
* **And** GPU memory usage should be logged after loading.

* **Given** the LLM (`google/gemma-3-27b-it`) *has* already been loaded and exists in the shared cache (`_model_cache`).
* **When** `load_shared_model` is called again with the same model name and device.
* **Then** the script should log that it is using the cached model.
* **And** the model and tokenizer should be retrieved directly from the `_model_cache`.
* **And** the cached model and tokenizer should be returned without reloading from Hugging Face.

### Scenario: Interfacing with Autogen Framework

* **Given** an Autogen agent requires an LLM client configuration.
* **And** the configuration specifies `model_client_cls` as `CustomGemmaClient`.
* **When** Autogen instantiates the client using this configuration.
* **Then** the `CustomGemmaClient.__init__` method should be called.
* **And** it should trigger `load_shared_model` to get the LLM instance.
* **When** the Autogen agent calls the client's `create` method with a list of messages.
* **Then** the `CustomGemmaClient._format_chat_prompt` method should format the messages into the Gemma-specific template.
* **And** the `CustomGemmaClient.create` method should use the loaded model to generate a response based on the formatted prompt and configured parameters.
* **And** the response should be returned in a structure compatible with Autogen (using `SimpleNamespace`).

### Scenario: Cleaning LLM Response Artifacts

* **Given** the LLM generates a raw response string containing Gemma chat template artifacts (e.g., `<start_of_turn>model\nRewritten text here.<end_of_turn>`).
* **When** the `clean_response` function is called with this raw string.
* **Then** the function should remove the `<start_of_turn>model\n` prefix and any `<end_of_turn>` suffix.
* **And** it should return only the core generated content ("Rewritten text here.").

### Scenario: Rewriting Text from a JSON File (Standalone Use)

* **Given** a JSON file exists (e.g., `output_from_step3.json`) containing `original_text` and moderator `improvements`.
* **And** the shared LLM is loaded.
* **When** the `rewrite_function.py` script is executed from the command line with the JSON file path as an argument (`python rewrite_function.py output_from_step3.json`).
* **Then** the `main` function should call `rewrite_text_from_json`.
* **And** `rewrite_text_from_json` should call `extract_from_json` to read the data.
* **And** `rewrite_text_from_json` should call `rewrite_text` with the extracted data.
* **And** the resulting rewritten text should be printed to the standard output (console).

## Key Components Involved (Behavioral Roles)

* **`rewrite_text`:** The primary service function that takes original text and feedback, orchestrates the LLM interaction via `call_model`, cleans the result, and returns the improved text. Handles retries.
* **`load_shared_model`:** Manages the loading and caching of the potentially large LLM to ensure it's only loaded once per process, handling quantization and memory settings.
* **`CustomGemmaClient`:** Acts as an adapter to make the loaded Gemma model compatible with the Autogen framework used in `step3.py`, handling prompt formatting specific to Gemma.
* **`call_model`:** A lower-level wrapper for invoking the `CustomGemmaClient` outside the main Autogen agent loop, used directly by `rewrite_text`.
* **`create_rewrite_prompt`:** Responsible for constructing the specific instructions and context sent to the LLM to guide the rewriting task.
* **`clean_response`:** Post-processes the raw LLM output to remove model-specific artifacts and extract the desired rewritten content.
* **`extract_from_json` / `rewrite_text_from_json` / `main`:** Provide the capability to run the rewrite function standalone on saved JSON files, primarily for testing or batch processing.

## Inputs and Outputs (Behavioral)

* **Input (Primary Use via `rewrite_text`):**
  * `original_text` (string).
  * `improvement_points` (list of strings).
  * Optional context: `referenced_papers` (dict), `full_chapter_context` (string).
  * Relies on a loaded LLM instance (via `load_shared_model`).
* **Output (Primary Use via `rewrite_text`):**
  * Rewritten text (string).
* **Input (Standalone Use via `main`):**
  * Path to a JSON file containing original text and improvements.
* **Output (Standalone Use via `main`):**
  * Rewritten text printed to standard output.
  * Log messages detailing the process.

## Interactions with Other Components

* **File System:** Reads `.env`; Reads JSON files (when run standalone); Writes logs (to console); Potentially writes model offload files to `./offload/`.
* **`step3.py`:** Imports and calls `rewrite_text`, `load_shared_model`, and uses `CustomGemmaClient` in its agent configuration.
* **LLM (Gemma 27B via `load_shared_model`):** The core engine used for performing the text rewriting.
* **Transformers/Torch Libraries:** Used for loading the LLM, tokenization, generation, and managing GPU resources.
* **Logging Module:** Used to report progress, errors, and memory usage.
* **Autogen Library:** `CustomGemmaClient` is designed to interface with this framework when called from `step3.py`.
