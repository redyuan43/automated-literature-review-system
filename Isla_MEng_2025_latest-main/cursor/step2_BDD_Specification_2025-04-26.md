# BDD Specification: step2.py

**Date:** 2025-04-26

## Feature: Initial Report Chapter Generation

As a user running the report generation pipeline,
I want to generate initial drafts of report chapters based on a topic or question,
So that these drafts can be reviewed and refined in the next step.

## Key Scenarios

### Scenario: Generating a Full Set of Chapters for a Topic

* **Given** a vector store and metadata exist in the `./embeddings` directory (created by `step1.py`).
* **And** a large language model (Gemma 27B) is accessible and configured.
* **And** the output directory `./initial_chapters` exists or can be created.
* **And** network access is available for potential CrossRef API calls.
* **When** the `step2.py` script is executed.
* **Then** the script should first generate a set of research questions related to the main topic ("semiconductors").
* **And** for each generated research question, the script should generate a corresponding chapter.
* **And** each chapter generation should involve:
  * Retrieving relevant context papers from the vector store based on the question.
  * Generating sections: "Background", "Current Research", "Recommendations" using the LLM and the retrieved context.
  * Attempting to fetch enhanced citation details from CrossRef for the referenced papers.
  * Formatting a "References" section based on the referenced papers and CrossRef data.
  * Checking for and potentially rewriting sections if they are too similar to previously generated content within the same run.
* **And** a JSON file (e.g., `chapter_1.json`, `chapter_2.json`, ...) should be created in the `./initial_chapters` directory for each successfully generated chapter.
* **And** each JSON file should contain the research question, generated sections (Background, Current Research, Recommendations, References), and metadata about the referenced papers.
* **And** log messages should indicate the progress of question generation and chapter generation, including any retries or errors encountered.

### Scenario: Retrieving Context for a Research Question

* **Given** a valid FAISS vector store (`index.faiss` or `faiss.index`) and metadata (`metadata.npy`) exist in `./embeddings` or `./converted_index`.
* **And** a specific research question string (e.g., "What are the latest advancements in GaN semiconductors?").
* **When** context retrieval is triggered for this question (within `generate_report` calling `get_paper_context`).
* **Then** the vector store should be queried with the research question.
* **And** a list of relevant document chunks should be returned.
* **And** these chunks should be grouped by their original paper title.
* **And** the top N papers (based on relevance scores, default N=8) should be selected.
* **And** the text content from the chunks of these top papers should be combined into a single context string.
* **And** metadata for these top N papers should be returned.

### Scenario: Generating a Single Report Section

* **Given** a specific research question (e.g., "Impact of EUV lithography on semiconductor scaling").
* **And** a context string containing relevant information retrieved from the vector store.
* **And** a target section name (e.g., "Background").
* **When** the LLM generation is triggered for this section (within `generate_report` calling `call_model`).
* **Then** a prompt should be constructed containing the research question, the context string, and instructions to generate the specified section.
* **And** the `CustomGemmaClient` should send this prompt to the configured Gemma LLM.
* **And** the LLM should return generated text content for the "Background" section.
* **And** this generated text should be cleaned of model artifacts.

### Scenario: Enhancing Citations with CrossRef

* **Given** a list of papers retrieved from the vector store, including their titles.
* **And** network access to `api.crossref.org` is available.
* **When** the citation organization process occurs (within `generate_report` calling `organize_papers_for_citation` which might call `get_crossref_citation`).
* **Then** for each paper title, the CrossRef API should be queried.
* **And** if a matching record is found on CrossRef, structured citation data (DOI, authors, year, journal) should be retrieved and associated with that paper.
* **And** the final "References" section should prioritize using the CrossRef data for formatting citations where available.

### Scenario: Detecting and Rewriting Repetitive Content

* **Given** a `ContentEmbeddingStore` is tracking generated sections.
* **And** a new section (e.g., "Recommendations") is generated.
* **When** the repetition check is performed (within `generate_report` calling `check_section_repetition`).
* **Then** the new section's content and summary should be compared against the content/summaries already in the `ContentEmbeddingStore`.
* **And** if the similarity score exceeds a predefined threshold (e.g., 0.85), the section should be identified as repetitive.
* **And** a rewrite process should be triggered (`rewrite_section`).
* **And** the LLM should be prompted to rewrite the section to be distinct from the similar content previously generated.
* **And** the report should be updated with the rewritten section content.
* **And** the new (rewritten) section content should be added to the `ContentEmbeddingStore`.

## Key Components Involved (Behavioral Roles)

* **`main`:** Orchestrates the overall process: generates questions, loops through questions, triggers report generation for each, and saves the results. Manages GPU memory between chapters.
* **`generate_research_questions`:** Interacts with the LLM to brainstorm relevant and diverse research questions based on a starting topic.
* **`generate_report`:** Manages the generation of a single chapter/report for a given question. It coordinates context retrieval, section generation, citation handling, and repetition checking.
* **`load_vector_store`:** Loads the pre-computed vector index and metadata needed for context retrieval.
* **`get_paper_context`:** Queries the vector store to find and return relevant text passages and paper metadata based on the input question/topic.
* **`call_model` / `CustomGemmaClient`:** Handles the interaction with the underlying Gemma LLM, including prompt formatting and response cleaning, to generate text for questions and report sections.
* **`organize_papers_for_citation` / `get_crossref_citation` / `format_reference_section`:** Work together to clean paper metadata, query CrossRef for better citation details, assign citation IDs, and format the final reference list.
* **`ContentEmbeddingStore` / `check_section_repetition` / `rewrite_section`:** Cooperate to detect, rewrite, and track generated content to reduce redundancy.

## Inputs and Outputs (Behavioral)

* **Input:**
  * FAISS index and metadata files from `./embeddings/` (output of `step1.py`).
  * A starting topic (hardcoded as "semiconductors" in `main`).
  * Configuration for the Gemma LLM (via environment variables/`OAI_CONFIG_LIST`).
  * (Potentially) Hugging Face token from `.env`.
* **Output:**
  * Multiple JSON files (`chapter_*.json`) saved in `./initial_chapters/`, each representing a generated report chapter with its question, sections, and reference metadata.
  * Log output detailing the process, including questions generated, context retrieved, sections generated, CrossRef lookups, repetition checks, and any errors.

## Interactions with Other Components

* **File System:** Reads index/metadata from `./embeddings/`; Reads `.env`; Writes chapter JSON files to `./initial_chapters/`; Writes potential offload files to `./offload/`; Reads/writes cache files to temp directory.
* **`step1.py` Output:** Directly consumes the FAISS index and metadata generated by `step1.py`.
* **LLM (Gemma 27B via `CustomGemmaClient` / `rewrite_function.py`'s `load_shared_model`):** Used extensively for generating research questions, report sections (Background, Current Research, Recommendations), summaries (for repetition check), and potentially rewriting sections.
* **SentenceTransformer Library:** Used by `ContentEmbeddingStore` to embed generated text for similarity comparison.
* **Langchain / FAISS Libraries:** Used to load and query the vector store (`load_vector_store`, `get_paper_context`).
* **CrossRef API:** External web service queried by `get_crossref_citation` to fetch citation details.
* **Logging Module:** Used to record execution progress and errors.
