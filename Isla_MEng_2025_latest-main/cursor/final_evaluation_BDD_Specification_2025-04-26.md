# BDD Specification: final_evaluation.py

**Date:** 2025-04-26

## Feature: Assessing Quality of Generated Text

As a user or system component (like `step3.py`),
I want to evaluate the quality of generated text sections based on multiple criteria,
So that I can quantitatively measure technical depth, clarity, structure, and citation accuracy.

## Key Scenarios

### Scenario: Evaluating Technical Depth of a Text Section

* **Given** a text string representing a report section.
* **And** the OpenAI API key is configured correctly (or a fallback mechanism is in place).
* **And** a dictionary of relevant technical terms is defined.
* **And** a spaCy NLP model (`en_core_web_sm`) is loaded.
* **When** the `calculate_technical_depth` function is called with the text.
* **Then** the function should identify technical terms from the predefined dictionary within the text.
* **And** it should use NLP (spaCy NER) to identify additional potential technical entities.
* **And** it should analyze the syntactic complexity of the sentences using dependency parsing.
* **And** it should call the OpenAI API (GPT-4) to get a qualitative assessment and score for technical depth, using a critical prompt.
* **And** it should combine these measures (term frequency, NER results, syntactic complexity, LLM score) into a final normalized technical depth score between 0 and 1.
* **And** the function should return a dictionary containing the final `score` and detailed `components` (dictionary score, NER score, syntax score, LLM evaluation details).

### Scenario: Evaluating Clarity of a Text Section

* **Given** a text string representing a report section.
* **And** the `textstat` library is available.
* **And** the OpenAI API key is configured correctly.
* **When** the `calculate_clarity` function is called with the text.
* **Then** the Gunning Fog index should be calculated for the text using `textstat`.
* **And** the calculated Gunning Fog index should be normalized to a score between 0 (very complex) and 1 (very simple), with an optimal range targeted.
* **And** the function should call the OpenAI API (GPT-4) to get a qualitative assessment and score for clarity, focusing on readability, flow, and conciseness.
* **And** the final clarity score should be a weighted average of the normalized Gunning Fog score and the LLM clarity score.
* **And** the function should return a dictionary containing the final `score` and detailed `components` (Gunning Fog score, LLM evaluation details).

### Scenario: Evaluating Structure of a Text Section

* **Given** a text string representing a report section.
* **And** an `sklearn` environment is available for text vectorization and topic modeling (LDA).
* **And** the OpenAI API key is configured correctly.
* **When** the `calculate_structure` function is called with the text.
* **Then** the text should be analyzed for contextual coherence using sentence embeddings (via `ContextualCoherenceAnalyzer`).
* **And** topic hierarchy and consistency should be analyzed using LDA.
* **And** the function should call the OpenAI API (GPT-4) to get a qualitative assessment and score for structure, focusing on logical flow, section organization, and transitions.
* **And** the final structure score should be a combination of the coherence score, topic analysis score, and the LLM structure score.
* **And** the function should return a dictionary containing the final `score` and detailed `components` (coherence score, topic score, LLM evaluation details).

### Scenario: Evaluating Citation Accuracy

* **Given** a text string representing a report section containing inline citations (e.g., "[1]", "[2]").
* **And** a dictionary `referenced_papers` mapping cleaned paper titles to their details, including the assigned `citation_id` (e.g., "[1]").
* **And** the OpenAI API key is configured correctly.
* **When** the `evaluate_citation_accuracy` function is called with the text and the `referenced_papers` dictionary.
* **Then** the function should identify all inline citation markers in the text.
* **And** it should check if each found citation corresponds to a valid `citation_id` in the `referenced_papers`.
* **And** it should identify any missing or extraneous citations.
* **And** it should call the OpenAI API (GPT-4) with the text, the list of expected references, and the found citations to assess the appropriateness and accuracy of the citations in context.
* **And** the function should calculate scores for `completeness` (all expected citations are present) and `correctness` (no unexpected citations are present).
* **And** it should combine these scores with the LLM's assessment score into a final citation accuracy score.
* **And** the function should return a dictionary containing the final `score`, detailed `components` (completeness, correctness, LLM evaluation), and lists of `found_citations`, `expected_citations`, `missing_citations`, and `unexpected_citations`.

### Scenario: Handling OpenAI API Failure During Evaluation

* **Given** a text string needs evaluation (e.g., for technical depth).
* **And** the OpenAI API key is invalid or network connectivity fails.
* **When** an evaluation function (e.g., `evaluate_technical_depth_with_llm`) attempts to call the OpenAI API.
* **Then** the API call should fail.
* **And** the function should catch the exception.
* **And** a warning message should be printed or logged indicating the LLM evaluation failed.
* **And** the function should return a default score (e.g., 0.5) and a justification indicating the failure.
* **And** the overall calculation function (e.g., `calculate_technical_depth`) should use this default score in its final calculation, but should still return successfully with scores from other available components (e.g., dictionary, NER, syntax).

## Key Components Involved (Behavioral Roles)

* **`calculate_technical_depth`:** Assesses how technically sophisticated and accurate the text is, using a combination of term dictionaries, NLP analysis, and LLM judgment.
* **`calculate_clarity`:** Measures how clear, readable, and concise the text is, using readability formulas (`textstat`) and LLM judgment.
* **`calculate_structure`:** Evaluates the logical flow, coherence, and organization of the text using sentence embeddings, topic modeling, and LLM judgment.
* **`evaluate_citation_accuracy`:** Checks if the inline citations in the text correctly and completely match the provided list of references, also using LLM judgment for contextual appropriateness.
* **LLM Evaluation Functions (e.g., `evaluate_technical_depth_with_llm`, `evaluate_clarity_with_llm`, etc.):** Internal helpers that interact directly with the OpenAI API (GPT-4) to get qualitative assessments for specific quality dimensions using carefully crafted prompts.
* **NLP/Utility Functions (e.g., `analyze_sentence_complexity_normalized`, `normalize_gunning_fog`, `analyze_topic_hierarchy_normalized`, `ContextualCoherenceAnalyzer`):** Provide specific metrics based on linguistic analysis (syntax, readability, coherence, topics) used as components in the main evaluation functions.

## Inputs and Outputs (Behavioral)

* **Input (Primary Functions):**
  * `text` (string): The text content of the section to be evaluated.
  * `referenced_papers` (dict): (For `evaluate_citation_accuracy` only) A dictionary mapping paper titles to their metadata including citation IDs.
  * Relies on a configured OpenAI API key (via `.env` or environment variable).
  * Relies on loaded spaCy model and potentially other NLP resources.
* **Output (Primary Functions):**
  * A dictionary containing:
    * `score` (float): The final normalized quality score (0-1) for the specific dimension (depth, clarity, structure, citation).
    * `components` (dict): A breakdown of the scores from different methods used in the calculation (e.g., dictionary score, syntax score, LLM score, Gunning Fog, coherence score, completeness, correctness).
    * (For citation accuracy) Lists detailing found, expected, missing, and unexpected citations.
    * (For LLM components) Justification text provided by the LLM.

## Interactions with Other Components

* **`step3.py`:** Imports and calls `calculate_technical_depth`, `calculate_clarity`, `calculate_structure`, and `evaluate_citation_accuracy` to assess the quality of original and rewritten text sections.
* **OpenAI API:** Used extensively by the LLM evaluation helper functions to get qualitative scores and justifications via GPT-4 model calls. Requires a valid API key.
* **spaCy Library:** Used for natural language processing tasks like sentence boundary detection, dependency parsing (`analyze_sentence_complexity_normalized`), and potentially named entity recognition (NER) if implemented within technical depth calculation. Requires a model like `en_core_web_sm`.
* **Textstat Library:** Used to calculate the Gunning Fog readability index (`calculate_clarity`).
* **Scikit-learn Library:** Used for text vectorization (`CountVectorizer`) and topic modeling (`LatentDirichletAllocation`) within `analyze_topic_hierarchy_normalized` (part of `calculate_structure`).
* **dotenv Library:** Used to load the OpenAI API key from a `.env` file.
* **File System:** Reads the `.env` file.
