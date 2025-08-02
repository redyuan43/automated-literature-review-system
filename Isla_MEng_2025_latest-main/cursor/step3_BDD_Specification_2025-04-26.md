# BDD Specification: step3.py

**Date:** 2025-04-26

## Feature: AI-Assisted Review and Improvement of Report Chapters

As a user running the report generation pipeline,
I want to automatically review and improve the initial chapter drafts using specialized AI agents,
So that the final chapters meet quality standards for technical accuracy, clarity, structure, and citation correctness.

## Key Scenarios

### Scenario: Processing All Initial Chapters

* **Given** one or more chapter JSON files (e.g., `chapter_1.json`) exist in the `./initial_chapters` directory (output from `step2.py`).
* **And** the required AI agents (Technical Accuracy, Clarity, Structure, Fact Checking, Moderator) are configured and operational (using `CustomGemmaClient`).
* **And** the text rewriting function (`rewrite_text` from `rewrite_function.py`) is available.
* **And** the quality assessment functions (from `final_evaluation.py`) are available.
* **And** output directories (`./outputs/consolidated/`, `./chapter_markdowns/`, `./logs/`) exist or can be created.
* **When** the `step3.py` script is executed.
* **Then** the script should iterate through each chapter JSON file found in `./initial_chapters`.
* **And** for each chapter, it should first assess the initial quality of its main sections ("Background", "Current Research", "Recommendations").
* **And** based on the assessment, it should selectively trigger reviews by the relevant agents (e.g., if clarity is low, ClarityAgent is triggered).
* **And** if reviews are triggered, the feedback should be consolidated by the ModeratorAgent.
* **And** the section text should be rewritten based on the consolidated feedback using the rewrite function.
* **And** the quality of the rewritten section should be assessed again.
* **And** detailed results for each reviewed/rewritten section should be saved to a JSON file in `./outputs/consolidated/{chapter_num}/`.
* **And** a final, improved chapter Markdown file should be created in `./chapter_markdowns/`.
* **And** final quality scores for the entire chapter should be saved to a JSON file in `./outputs/`.
* **And** log messages should detail the process for each chapter and section, including initial scores, agents triggered, feedback received, rewrite attempts, and final scores.

### Scenario: Section Quality Meets Thresholds (No Review Needed)

* **Given** a chapter JSON file exists in `./initial_chapters`.
* **And** the initial quality assessment (`assess_quality`) determines that a specific section (e.g., "Background") meets all predefined quality thresholds (technical depth, clarity, structure, citation accuracy).
* **When** the `step3.py` script processes this section within its chapter loop.
* **Then** the `check_metric_thresholds` function should indicate that no improvements are needed for this section.
* **And** the `get_needed_agents` function should return an empty list.
* **And** the `selective_review_section` function should *not* trigger any review agents (Technical, Clarity, Structure, FactChecking) or the ModeratorAgent for this section.
* **And** the section content should *not* be sent to the `rewrite_text` function.
* **And** the original section content should be used directly in the final chapter Markdown output.
* **And** log messages should indicate that the section passed initial checks and no review/rewrite was performed.

### Scenario: Section Quality Requires Specific Improvement (e.g., Clarity)

* **Given** a chapter JSON file exists in `./initial_chapters`.
* **And** the initial quality assessment (`assess_quality`) determines that the "Current Research" section scores below the threshold for clarity, but meets thresholds for technical accuracy, structure, and citations.
* **When** `step3.py` processes this section via `selective_review_section`.
* **Then** the `check_metric_thresholds` function should indicate a need for clarity improvement.
* **And** the `get_needed_agents` function should return a list containing at least the 'ClarityAgent'.
* **And** the `ClarityAgent` should be invoked to review the section content.
* **And** the `ModeratorAgent` should be invoked to consolidate the feedback (which primarily comes from the ClarityAgent in this case).
* **And** the consolidated feedback (focused on clarity) should be extracted.
* **And** the `rewrite_text` function should be called with the original text and the clarity-focused feedback.
* **Then** the rewritten text should be generated.
* **And** the quality of the rewritten text should be assessed again.
* **And** the results (original, feedback, rewritten text, scores) should be saved to the consolidated output JSON for that section.
* **And** the rewritten text should be used in the final chapter Markdown.

### Scenario: Agent Provides Formatted Feedback

* **Given** a specific review agent (e.g., `TechnicalAccuracyAgent`) is invoked to review a section's content.
* **When** the agent processes the review request via its `review` method and the underlying `CustomGemmaClient` interacts with the LLM.
* **Then** the agent should return a response containing feedback.
* **And** this feedback should ideally be presented as a numbered list enclosed within `***` markers, as specified in the agent's system message (e.g., "***\n1. Point one.\n2. Point two.\n***").
* **And** helper functions (`extract_last_asterisk_section`, `parse_improvement_points`) should be able to successfully extract this numbered list from the agent's raw response string.

### Scenario: Moderator Consolidates Feedback

* **Given** multiple review agents (e.g., TechnicalAccuracyAgent, ClarityAgent) have provided feedback on the same section.
* **When** the `ModeratorAgent` is invoked via `selective_review_section` with the section content and the collected reviews.
* **Then** the `ModeratorAgent` should process the input reviews.
* **And** it should generate a single, consolidated, and potentially prioritized list of actionable improvement points.
* **And** this consolidated list should be formatted within `***` markers in its response.
* **And** the `get_moderator_improvements` function should successfully extract this consolidated list.

## Key Components Involved (Behavioral Roles)

* **`main`:** Orchestrates the entire review process, looping through chapters and sections, coordinating assessments, reviews, rewrites, and saving outputs.
* **`assess_quality`:** Evaluates the quality of a given text section based on predefined metrics (delegates to functions in `final_evaluation.py`).
* **`check_metric_thresholds` / `get_needed_agents`:** Determine *if* a section needs review and *which* specific aspects (and thus agents) require attention based on the initial assessment.
* **`selective_review_section`:** Manages the review lifecycle for a single section based on the needed agents, including agent invocation, moderation, rewriting, and re-assessment.
* **Review Agents (`TechnicalAccuracyAgent`, `ClarityAgent`, `StructureAgent`, `FactCheckingAgent`):** Each agent acts as a specialist, reviewing the text from its specific perspective and providing targeted feedback in a structured format.
* **`ModeratorAgent`:** Synthesizes feedback from multiple review agents into a single, actionable list for the rewriting step.
* **`rewrite_text` (from `rewrite_function.py`):** Performs the actual text modification based on the consolidated feedback provided by the ModeratorAgent.
* **`CustomGemmaClient` (from `rewrite_function.py`):** Provides the underlying interface for all agents to interact with the Gemma LLM.
* **`save_consolidated_output`:** Saves detailed intermediate results (original text, feedback, rewritten text, scores) for each reviewed section.
* **`create_chapter_markdown`:** Assembles the final, potentially improved sections into a complete chapter Markdown file.

## Inputs and Outputs (Behavioral)

* **Input:**
  * Chapter JSON files from `./initial_chapters/` (output of `step2.py`).
  * Configuration for AI agents and the LLM (via `OAI_CONFIG_LIST`).
  * Quality metric functions (from `final_evaluation.py`).
  * Text rewriting function (from `rewrite_function.py`).
  * (Potentially) Hugging Face token from `.env`.
* **Output:**
  * Improved chapter Markdown files saved in `./chapter_markdowns/`.
  * Detailed JSON logs for each reviewed/rewritten section saved in `./outputs/consolidated/{chapter_num}/`.
  * Final quality assessment scores for each chapter saved in `./outputs/`.
  * Execution logs saved in `./logs/improvement_process_{timestamp}.log`.

## Interactions with Other Components

* **File System:** Reads chapter JSONs from `./initial_chapters/`; Reads `.env`; Writes detailed JSONs to `./outputs/consolidated/`; Writes final assessment JSONs to `./outputs/`; Writes Markdown files to `./chapter_markdowns/`; Writes logs to `./logs/`.
* **`step2.py` Output:** Directly consumes the chapter JSON files generated by `step2.py`.
* **`rewrite_function.py`:** Imports and uses `rewrite_text`, `load_shared_model`, and `CustomGemmaClient` for LLM interaction and text modification.
* **`final_evaluation.py`:** Imports and uses functions like `calculate_technical_depth`, `calculate_clarity`, etc., for quality assessment.
* **Autogen Library:** Used as the framework for defining and coordinating the multi-agent review process.
* **Transformers/Torch Libraries:** Used implicitly via `CustomGemmaClient` and `rewrite_function.py` for LLM operations and GPU management.
* **Logging Module:** Used extensively to record the multi-step review and rewrite process.
