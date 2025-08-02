# BDD Specification: Whole-system-Masters-Project

**Date:** 2025-04-26

## 1. Project Overview & Core Purpose

This project aims to automate the creation, refinement, and evaluation of technical academic reports or literature reviews. It addresses the challenge of synthesizing information from multiple source documents and iteratively improving the generated content using AI agents and defined quality metrics. The core purpose is to produce high-quality, contextually relevant reports based on a specified topic and source materials, leveraging LLMs and vector search capabilities within an HPC environment.

## 2. High-Level Features

* **F1: Document Indexing:** Creates a searchable vector knowledge base from a collection of source documents (research papers).
* **F2: Context-Aware Report Generation:** Generates initial sections of an academic report based on a given topic or research question, using relevant context retrieved from the indexed documents.
* **F3: Automated Report Review:** Utilizes a team of specialized AI agents to review generated report sections based on criteria like technical accuracy, clarity, structure, and fact-checking.
* **F4: Iterative Report Rewriting:** Refines and rewrites report sections based on the consolidated feedback gathered during the automated review process.
* **F5: Concept Visualization:** Generates visual diagrams (Mermaid format) representing the key concepts discussed in the final reports.
* **F6: Quality Evaluation:** Assesses the quality of generated or rewritten reports using a suite of metrics including readability, technical depth, structure, and citation accuracy.
* **F7: Domain-Specific Model Fine-Tuning (Optional):** Fine-tunes a base language model using the final generated reports to potentially improve performance on domain-specific tasks.
* **F8: Orchestrated Pipeline Execution:** Executes the entire workflow (indexing, generation, review, etc.) sequentially via a batch script suitable for an HPC/SLURM environment.

## 3. Detailed Feature Scenarios

### Feature: F1: Document Indexing

* **Scenario:** Creating the knowledge base from source papers
  * **Given:** A directory (`./files_mmd`) containing source documents in Markdown format.
  * **When:** The indexing script (`step1.py`) is executed (e.g., via `batch_script.sh`).
  * **Then:** Metadata (title, authors, etc.) should be extracted from the documents.
  * **And:** The content of each document should be split into overlapping chunks.
  * **And:** Vector embeddings should be generated for each chunk using a sentence transformer model.
  * **And:** A FAISS index containing these embeddings should be created and saved to the embeddings directory (`./embeddings/faiss.index`).
  * **And:** Associated metadata for each chunk should be saved alongside the index (`./embeddings/metadata.*`).
  * **And:** Statistics about the indexing process should be saved (`./embeddings/stats.json`).
  * **And:** Progress should be logged to the console and `document_indexing.log`.

### Feature: F2: Context-Aware Report Generation

* **Scenario:** Generating initial report sections for a research question
  * **Given:** A FAISS index and metadata exist in the embeddings directory (`./embeddings`).
  * **And:** A research topic or specific question (e.g., "challenges in semiconductor scaling") is provided.
  * **When:** The report generation script (`step2.py`) is executed.
  * **Then:** The script should load the FAISS vector store.
  * **And:** It should perform a similarity search using the topic/question to retrieve relevant document chunks.
  * **And:** It should format a prompt for the LLM (Gemma), including the retrieved context.
  * **And:** It should call the LLM (via `CustomGemmaClient`) to generate content for predefined report sections (e.g., Background, Current Research, Recommendations).
  * **And:** It may attempt to fetch citation details from CrossRef for identified papers.
  * **And:** The generated report, including sections and referenced paper metadata, should be saved as a structured JSON file in the initial chapters directory (`./initial_chapters`).
  * **And:** GPU memory usage should be monitored and managed.

### Feature: F3: Automated Report Review

* **Scenario:** Reviewing a generated report section using AI agents
  * **Given:** An initial report section exists (e.g., from a JSON file in `./initial_chapters`).
  * **And:** Referenced paper data for the section is available.
  * **When:** The review script (`step3.py`) processes the section.
  * **Then:** Specialized agents (TechnicalAccuracy, Clarity, Structure, FactChecking) should be invoked to review the section content.
  * **And:** Each agent should provide feedback as a numbered list of improvement points focused on its specialty.
  * **And:** The FactChecking agent should compare claims against the provided referenced paper data.
  * **And:** A Moderator agent should synthesize the feedback from all specialist agents into a single consolidated list, prioritizing fact-checking issues.
  * **And:** The raw reviews and the consolidated moderator feedback should be saved (likely to a JSON file in `./outputs`).

### Feature: F4: Iterative Report Rewriting

* **Scenario:** Improving a report section based on review feedback
  * **Given:** A report section and a consolidated list of improvement points (e.g., from the Moderator agent in `./outputs`).
  * **When:** The rewriting process is triggered (likely within `step3.py`, calling `rewrite_function.py`).
  * **Then:** A prompt should be constructed containing the original text and the improvement points.
  * **And:** The LLM (Gemma, via `CustomGemmaClient` in `rewrite_function.py`) should be called with the prompt.
  * **And:** The LLM should generate a rewritten version of the text attempting to address the feedback.
  * **And:** The rewritten text should preserve original citation markers (e.g., `[1]`).
  * **And:** The rewritten text should be returned or saved, potentially overwriting the previous version or being stored as part of an iteration history.

### Feature: F5: Concept Visualization

* **Scenario:** Generating a concept diagram for a final report
  * **Given:** A final, improved report exists in Markdown format (`./chapter_markdowns`).
  * **When:** The diagram generation script (`step4.py`) is executed.
  * **Then:** The script should read the markdown report content.
  * **And:** It should prompt the LLM (Gemma) to identify key concepts and their relationships.
  * **And:** The LLM should generate a response containing Mermaid diagram syntax.
  * **And:** The script should extract and validate the Mermaid code.
  * **And:** A new Markdown file containing the Mermaid diagram should be saved (e.g., in `./chapter_diagrams`).

### Feature: F6: Quality Evaluation

* **Scenario:** Assessing the quality of a generated/rewritten report section
  * **Given:** Text content of a report section.
  * **And:** (Optionally) Referenced paper data for citation checking.
  * **When:** Evaluation functions from `final_evaluation.py` are called (e.g., by `step3.py`).
  * **Then:** Metrics for technical depth should be calculated (using dictionaries, NLP, and potentially OpenAI API).
  * **And:** Metrics for clarity should be calculated (using readability scores, coherence analysis, and potentially OpenAI API).
  * **And:** Metrics for structure should be calculated (using topic modeling (LDA) and potentially OpenAI API).
  * **And:** If references are provided, citation accuracy should be evaluated (likely using OpenAI API).
  * **And:** A dictionary containing these metric scores and justifications should be returned.

### Feature: F7: Domain-Specific Model Fine-Tuning (Optional)

* **Scenario:** Fine-tuning a model on generated reports
  * **Given:** A collection of final reports in Markdown format (`./chapter_markdowns`).
  * **When:** The fine-tuning script (`fine_tune.py`) is executed.
  * **Then:** A base model (Mistral-7B) should be loaded with quantization.
  * **And:** The markdown reports should be loaded and prepared as a training dataset.
  * **And:** LoRA should be configured for parameter-efficient fine-tuning.
  * **And:** The model should be trained on the dataset using the Hugging Face Trainer.
  * **And:** The resulting fine-tuned model adapters should be saved (`./fine_tuned_model`).

### Feature: F8: Orchestrated Pipeline Execution

* **Scenario:** Running the end-to-end workflow on SLURM
  * **Given:** The project code, Conda environment (`new_env`), required `.env` file, and input data (`./files_mmd`) are set up on the HPC cluster.
  * **When:** The main batch script (`batch_script.sh`) is submitted to SLURM (`sbatch batch_script.sh`).
  * **Then:** The script should load the correct modules (CUDA 11.8) and activate the Conda environment.
  * **And:** It should execute `step1.py`, `step2.py`, `step3.py`, `step4.py`, and `fine_tune.py` in sequence (assuming all are uncommented).
  * **And:** Execution should stop if any script fails, logging an error.
  * **And:** Job output and errors should be logged to files specified in the script (e.g., `Isla_*.out`, `Isla_*.err`).
  * **And:** Email notifications should be sent based on job status (if configured).

## 4. Key Components Supporting Features

* **Document Indexing (F1):** `step1.py`, SentenceTransformer library, FAISS library.
* **Report Generation (F2):** `step2.py`, `CustomGemmaClient` (in `step2.py`/`rewrite_function.py`), FAISS/LangChain, Transformers library (Gemma model), CrossRef API.
* **Report Review (F3):** `step3.py`, Autogen library, specialized Agent classes (`TechnicalAccuracyAgent`, `ClarityAgent`, etc.), `CustomGemmaClient`.
* **Report Rewriting (F4):** `rewrite_function.py` (`rewrite_text` function), `CustomGemmaClient`, Transformers library (Gemma model). (Triggered by `step3.py`).
* **Concept Visualization (F5):** `step4.py`, `CustomGemmaClient`, Transformers library (Gemma model).
* **Quality Evaluation (F6):** `final_evaluation.py`, spaCy, textstat, OpenAI API, scikit-learn. (Called by `step3.py`).
* **Fine-Tuning (F7):** `fine_tune.py`, Transformers library (Mistral model, Trainer), PEFT library, Datasets library.
* **Pipeline Execution (F8):** `batch_script.sh` (or variants), SLURM workload manager, Conda environment manager.

## 5. Key Data/Entities (from a behavioral perspective)

* **Source Document:** Represents an input research paper (Markdown format in `./files_mmd`). Contains text content and implicit metadata.
* **Document Chunk:** A segment of text derived from a Source Document, used for embedding and retrieval.
* **Embedding Vector:** A numerical representation of a Document Chunk's meaning.
* **Index Metadata:** Information (title, authors, source file, chunk ID) linked to Embedding Vectors.
* **FAISS Index:** The data structure enabling efficient search over Embedding Vectors.
* **Research Topic/Question:** A string defining the subject for report generation.
* **Initial Report:** A structured representation (likely JSON in `./initial_chapters`) of the first draft report, containing sections and referenced paper info.
* **Review Feedback:** Structured comments and improvement suggestions generated by AI agents (JSON in `./outputs`).
* **Improved Report:** The refined version of the report after incorporating feedback (Markdown in `./chapter_markdowns`).
* **Concept Diagram:** A visual representation (Mermaid code in `./chapter_diagrams`) of key concepts in a report.
* **Quality Metrics:** Numerical scores and textual justifications assessing aspects like clarity, depth, structure, accuracy.
* **Fine-Tuned Model:** The resulting model adapters after training on generated reports (`./fine_tuned_model`).

## 6. External Interactions & APIs (behavioral focus)

* **Hugging Face Hub Interaction (F2, F4, F7, Rewriting):**
  * **Given:** A script requiring an LLM (Gemma, Mistral) or embedding model needs to run.
  * **When:** The script initializes the model (`from_pretrained`).
  * **Then:** It interacts with the Hugging Face Hub to download model weights and tokenizer configuration (requires `HUGGINGFACE_TOKEN`).
* **CrossRef API Interaction (F2):**
  * **Given:** `step2.py` identifies potentially relevant papers during context retrieval.
  * **When:** The `get_crossref_citation` function is called for a paper title.
  * **Then:** An HTTP GET request is made to the Crossref API (`api.crossref.org`).
  * **And:** Citation metadata (DOI, authors, year, journal) is potentially retrieved and associated with the paper.
* **OpenAI API Interaction (F6):**
  * **Given:** A report section needs quality evaluation for depth, clarity, structure, or citation accuracy.
  * **When:** An evaluation function in `final_evaluation.py` (e.g., `evaluate_technical_depth_with_llm`) is called.
  * **Then:** An API call is made to the OpenAI API (specifically targeting GPT-4).
  * **And:** A JSON response containing a score and justification is expected (requires `OPENAI_API_KEY`).

## 7. Dependencies (related to features)

* **Document Indexing (F1):** `sentence-transformers` (Embeddings), `faiss-cpu` (Vector Index), `numpy` (Numerical Ops).
* **Report Generation (F2):** `transformers`/`pytorch`/`bitsandbytes` (Gemma LLM Interaction), `langchain-community`/`faiss-cpu` (Vector Store Access), `requests` (CrossRef API).
* **Report Review (F3):** `pyautogen` (Agent Framework), `transformers`/`pytorch`/`bitsandbytes` (LLM for Agents).
* **Report Rewriting (F4):** `transformers`/`pytorch`/`bitsandbytes` (Gemma LLM Interaction - via `rewrite_function.py`).
* **Concept Visualization (F5):** `transformers`/`pytorch`/`bitsandbytes` (Gemma LLM Interaction).
* **Quality Evaluation (F6):** `openai` (GPT-4 API), `spacy` (NLP Analysis), `textstat` (Readability), `sklearn` (LDA Topic Modeling).
* **Fine-Tuning (F7):** `transformers`/`pytorch`/`bitsandbytes` (Mistral LLM), `peft` (LoRA), `datasets` (Data Handling).
* **Pipeline Execution (F8):** SLURM (Job Scheduling), Conda (Environment), Bash (Scripting).
* **General:** `python-dotenv` (Environment Variables), `logging` (Logging), `json`/`pyyaml` (Data Serialization).
