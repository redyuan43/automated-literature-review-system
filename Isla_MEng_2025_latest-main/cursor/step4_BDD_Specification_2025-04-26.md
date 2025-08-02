# BDD Specification: step4.py

**Date:** 2025-04-26

## Feature: Concept Diagram Generation for Report Chapters

As a user who has generated report chapters,
I want to automatically create a visual concept diagram for each chapter,
So that I have a graphical summary of the key topics and their relationships.

## Key Scenarios

### Scenario: Generating Concept Diagrams for All Chapters

* **Given** one or more finalized chapter Markdown files (`.md`) exist in the `./chapter_markdowns` directory (output from `step3.py`).
* **And** a large language model (Gemma 27B) is accessible and configured via `load_model`.
* **And** the output directory `./outputs/concept_diagrams` exists or can be created.
* **When** the `step4.py` script is executed.
* **Then** the script should process each Markdown file found in `./chapter_markdowns`.
* **And** for each valid Markdown file, it should generate a Mermaid concept diagram.
* **And** a corresponding `.mmd` file containing the Mermaid code should be saved in the `./outputs/concept_diagrams` directory (e.g., `Chapter 1 Title_concept_diagram.mmd`).
* **And** log messages (in `./logs/concept_diagrams_{timestamp}.log` and console) should indicate the progress and success or failure for each file processed.

### Scenario: Generating a Diagram for a Single Chapter

* **Given** a valid Markdown file `Chapter_1_Semiconductor_Trends.md` exists in `./chapter_markdowns` with sufficient content (>50 characters).
* **And** the LLM (Gemma 27B) is loaded and available.
* **When** the script processes `Chapter_1_Semiconductor_Trends.md` (via `process_markdown_file`).
* **Then** the content of the file should be read.
* **And** a prompt should be sent to the LLM asking for a Mermaid diagram summarizing the key concepts (5-10 concepts) and relationships from the text.
* **And** the LLM response should be processed to extract the Mermaid code block.
* **And** if valid Mermaid code is extracted, it should be saved to `./outputs/concept_diagrams/Chapter 1 Semiconductor Trends_concept_diagram.mmd`.
* **And** log messages should confirm the generation and saving of the diagram for this file.

### Scenario: Handling a Chapter File with Insufficient Content

* **Given** a Markdown file `empty_chapter.md` exists in `./chapter_markdowns` but contains less than 50 characters of text.
* **And** the LLM is loaded.
* **When** the script attempts to process `empty_chapter.md` (via `process_markdown_file`).
* **Then** the `read_markdown_file` function should identify the insufficient content.
* **And** a warning message should be logged indicating insufficient content for `empty_chapter.md`.
* **And** the `generate_concept_diagram` function should *not* be called for this file.
* **And** no corresponding `.mmd` file should be created in `./outputs/concept_diagrams`.

### Scenario: Handling LLM Failure During Diagram Generation

* **Given** a valid Markdown file `complex_chapter.md` exists in `./chapter_markdowns`.
* **And** the LLM is loaded.
* **And** when processing `complex_chapter.md`, the LLM fails to generate a response or generates an invalid/empty response.
* **When** the script processes `complex_chapter.md` and calls `generate_concept_diagram`.
* **Then** the attempt to generate or extract Mermaid code should fail.
* **And** an error or warning message indicating the failure to generate/extract the diagram for `complex_chapter.md` should be logged.
* **And** no `.mmd` file should be created in `./outputs/concept_diagrams` for this chapter.
* **And** the script should continue to process the next chapter file.

### Scenario: Correctly Extracting Mermaid Code from LLM Response

* **Given** the LLM generates a response for a chapter containing text, explanations, and a valid Mermaid code block enclosed in ```mermaid ...``` tags.
* **When** the `generate_concept_diagram` function receives this raw response.
* **Then** the function should attempt to extract the text specifically between the ```mermaid and``` tags.
* **And** the extracted text should be validated (e.g., contains expected keywords like `flowchart`, `graph`, `mindmap`).
* **And** only the validated Mermaid code block should be returned by the function, stripped of surrounding text and backticks.

## Key Components Involved (Behavioral Roles)

* **`main`:** Orchestrates the processing of all chapter files, loads the model, and manages the overall workflow including progress tracking and memory management.
* **`process_markdown_file`:** Handles the end-to-end process for a single Markdown file: reading, triggering diagram generation, and saving the output.
* **`read_markdown_file`:** Reads a specific Markdown file, extracts its title, and validates if it has enough content to proceed.
* **`generate_concept_diagram`:** Interacts with the LLM by constructing the appropriate prompt, sending the request, and robustly extracting the resulting Mermaid code from the response.
* **`load_model`:** Responsible for loading the specified LLM (Gemma 27B) and tokenizer, including handling quantization, memory configuration, and caching.
* **`setup_logging`:** Configures how progress and errors are reported during execution.

## Inputs and Outputs (Behavioral)

* **Input:**
  * Finalized chapter Markdown files (`.md`) located in the `./chapter_markdowns/` directory.
  * Configuration for the LLM (default model name, potentially quantization/memory settings).
  * (Potentially) Hugging Face token from `.env`.
* **Output:**
  * Mermaid diagram files (`.mmd`) saved in the `./outputs/concept_diagrams/` directory, named based on the input chapter titles.
  * Log files (`./logs/concept_diagrams_{timestamp}.log`) containing details of the execution, including successes, warnings (e.g., insufficient content, truncation), and errors (e.g., generation failures).

## Interactions with Other Components

* **File System:** Reads `.md` files from `./chapter_markdowns/`; Writes `.mmd` files to `./outputs/concept_diagrams/`; Writes logs to `./logs/`; Reads `.env`; Potentially writes offload files to `./offload/`.
* **`step3.py` Output:** Directly consumes the Markdown chapter files generated by `step3.py`.
* **LLM (Gemma 27B via `load_model`):** Used to analyze chapter content and generate Mermaid diagram code based on prompts.
* **Transformers/Torch Libraries:** Used for loading the LLM, tokenization, generation, and managing GPU resources.
* **Logging Module:** Used to record the progress and outcome of processing each file.
* **Tqdm Library:** Used to display a progress bar during the processing of multiple files.
