# Technical Specification: Whole-system-Masters-Project

**Date:** 2025-04-26

## 1. Project Overview

* **Project Name:** Whole-system-Masters-Project
* **Purpose:** This project implements an automated pipeline for generating, reviewing, refining, and evaluating academic literature reviews or technical reports on a given topic (e.g., semiconductors). It leverages Large Language Models (LLMs), vector databases, and agent-based frameworks to automate various stages of the research and writing process, culminating in structured reports, concept diagrams, and potentially a fine-tuned model specific to the domain.
* **Primary Goals:**
  * Automate the retrieval and synthesis of information from source documents.
  * Generate structured initial reports based on research questions.
  * Implement an iterative review process using specialized AI agents for feedback.
  * Rewrite reports based on consolidated feedback.
  * Evaluate the quality of generated content using objective and LLM-based metrics.
  * Visualize key concepts using diagrams.
  * Optionally fine-tune an LLM on the generated content.

## 2. Architecture

* **High-Level Architecture:** The system follows a **Sequential Pipeline Architecture**, orchestrated primarily through shell scripts (`batch_script*.sh`) designed for a SLURM HPC environment. Each step in the pipeline is typically handled by a dedicated Python script (`step1.py` to `step4.py`, `fine_tune.py`), processing data and passing outputs to the next stage via the file system.
* **Key Components/Stages:**
  * **Step 1: Indexing (`step1.py`):** Processes source documents (Markdown), extracts metadata, chunks content, generates vector embeddings, and creates a searchable FAISS index.
  * **Step 2: Initial Report Generation (`step2.py`):** Takes a topic/research question, queries the FAISS index for relevant context, interacts with an LLM (Gemma) to generate structured reports (JSON format), and potentially enriches citations via the CrossRef API.
  * **Step 3: Review & Refinement (`step3.py`):** Employs an Autogen-based multi-agent system (Technical, Clarity, Structure, Fact-Checking agents) to review the reports generated in Step 2. A Moderator agent consolidates feedback. This step likely invokes a rewriting process (using `rewrite_function.py`) based on the feedback and saves improved reports (Markdown) and detailed review outputs (JSON).
  * **Step 4: Diagram Generation (`step4.py`):** Takes the final markdown reports and uses an LLM (Gemma) to generate Mermaid concept diagrams visualizing key concepts.
  * **Step 5: Fine-Tuning (`fine_tune.py` - Optional):** Uses the final markdown reports as training data to fine-tune a separate LLM (Mistral) using PEFT/LoRA.
  * **Evaluation (`final_evaluation.py`):** Provides functions called by Step 3 (and potentially usable standalone) to assess report quality based on technical depth, clarity, structure, and citation accuracy using a mix of statistical methods, NLP, and external LLM APIs (GPT-4).
  * **Rewriting (`rewrite_function.py`):** Contains the core LLM interaction logic for rewriting text based on feedback points, likely called by Step 3.
* **Component Interaction:** Interaction is primarily file-based. `step1.py` produces embeddings/metadata read by `step2.py`. `step2.py` produces initial JSON reports read by `step3.py`. `step3.py` produces consolidated feedback (JSON) and final markdown reports. `step4.py` reads the final markdown reports. `fine_tune.py` reads the final markdown reports. `rewrite_function.py` is called by `step3.py`. `final_evaluation.py` functions are called by `step3.py`. The `batch_script.sh` orchestrates the sequential execution of these Python scripts.

## 3. Key Components/Modules

* **`step1.py` (Document Indexing):**
  * Responsibilities: Read source docs (`./files_mmd`), parse metadata, chunk text, embed chunks (SentenceTransformer), create FAISS index, save index/metadata (`./embeddings`).
  * Interfaces: Reads `.mmd` files, writes `.index`, `.npy`, `.json` files.
* **`step2.py` (Initial Report Generation):**
  * Responsibilities: Load FAISS index, perform similarity search, generate research questions (optional), format prompts with context, interact with Gemma LLM (via `CustomGemmaClient`) to generate report sections, handle citations (CrossRef API), check for repetition, save initial reports (`./initial_chapters`).
  * Interfaces: Reads FAISS index, interacts with Gemma LLM, CrossRef API, writes `.json` chapter files. Key class: `CustomGemmaClient`. Key function: `generate_report`.
* **`step3.py` (Review & Refinement):**
  * Responsibilities: Load initial reports, orchestrate Autogen review agents (Technical, Clarity, Structure, FactChecking, Moderator), consolidate feedback, trigger rewriting (via `rewrite_function.py`), assess quality (via `final_evaluation.py`), save review outputs (`./outputs`) and final markdown chapters (`./chapter_markdowns`).
  * Interfaces: Reads `.json` chapters, interacts with Autogen agents, calls `rewrite_text` and evaluation functions, writes `.json` feedback and `.md` reports. Key classes: `TechnicalAccuracyAgent`, `ClarityAgent`, `StructureAgent`, `FactCheckingAgent`, `ModeratorAgent`. Key functions: `selective_review_section`, `main` loop.
* **`step4.py` (Diagram Generation):**
  * Responsibilities: Read final markdown reports, format prompts, interact with Gemma LLM to generate Mermaid diagrams, save diagrams (`./chapter_diagrams`).
  * Interfaces: Reads `.md` files, interacts with Gemma LLM, writes `.md` diagram files. Key function: `generate_concept_diagram`.
* **`rewrite_function.py` (Rewriting Utilities):**
  * Responsibilities: Provide core `rewrite_text` function using Gemma LLM based on feedback, handle LLM loading/caching (`load_shared_model`), define LLM client (`CustomGemmaClient`), clean LLM responses.
  * Interfaces: Called by `step3.py`. Interacts with Gemma LLM. Can read input from `.json` files in `./outputs`.
* **`fine_tune.py` (LLM Fine-Tuning):**
  * Responsibilities: Load base model (Mistral), load training data (`./chapter_markdowns`), configure LoRA, run fine-tuning using Hugging Face Trainer, save fine-tuned model adapters (`./fine_tuned_model`).
  * Interfaces: Reads `.md` files, interacts with Hugging Face libraries, writes model files.
* **`final_evaluation.py` (Quality Assessment):**
  * Responsibilities: Provide functions to calculate metrics for technical depth, clarity, structure, and citation accuracy. Uses NLP (spaCy), statistical measures (textstat), and LLM APIs (OpenAI GPT-4).
  * Interfaces: Functions are imported and called by `step3.py`. Interacts with OpenAI API.

## 4. Data Management

* **Primary Data Structures:**
  * **Document Chunks:** Text segments from source documents.
  * **Embeddings:** Vector representations of document chunks (stored in FAISS).
  * **Metadata:** Information extracted from source documents (title, authors, etc.), stored alongside embeddings (`metadata.npy`, `metadata.json`).
  * **Report Sections:** Text content for different parts of the generated report (Background, Current Research, Recommendations, References).
  * **Consolidated Feedback:** JSON objects storing reviews and improvement points from agents (in `./outputs`).
  * **Referenced Papers:** Dictionary structure holding metadata and potentially content snippets of papers cited in a report.
* **Database Schema:** No traditional relational database is used. Data persistence relies on:
  * **FAISS Index (`embeddings/faiss.index`):** Stores vector embeddings for efficient similarity search.
  * **NumPy/JSON Files (`embeddings/metadata.*`):** Store metadata associated with indexed chunks.
  * **JSON Files (`initial_chapters/`, `outputs/`):** Store structured reports and review feedback.
  * **Markdown Files (`files_mmd/`, `chapter_markdowns/`, `chapter_diagrams/`):** Store source documents, final reports, and diagrams.
* **Data Flow:**
    1. Markdown files (`files_mmd/`) are read by `step1.py`.
    2. `step1.py` creates FAISS index and metadata in `embeddings/`.
    3. `step2.py` reads from `embeddings/`, generates initial reports saved as JSON in `initial_chapters/`.
    4. `step3.py` reads from `initial_chapters/`, processes reports, saves feedback JSON to `outputs/`, and writes final Markdown reports to `chapter_markdowns/`.
    5. `step4.py` reads from `chapter_markdowns/` and writes diagrams to `chapter_diagrams/`.
    6. `fine_tune.py` reads from `chapter_markdowns/` and writes model files to `fine_tuned_model/`.

## 5. APIs & Endpoints

* **Internal APIs:** No internal APIs are defined or exposed by the scripts themselves. Communication is file-based.
* **External APIs Consumed:**
  * **Hugging Face Hub:** Implicitly used by the `transformers` and `huggingface-hub` libraries to download models (Gemma, Mistral, SentenceTransformer) and tokenizers. Requires authentication (`HUGGINGFACE_TOKEN`).
  * **CrossRef API:** Explicitly called in `step2.py` (`get_crossref_citation`) to fetch citation metadata for retrieved papers. Uses `requests` library.
  * **OpenAI API:** Explicitly called in `final_evaluation.py` (`evaluate_*_with_llm` functions) to get GPT-4 based quality assessments. Requires authentication (`OPENAI_API_KEY`).

## 6. Core Algorithms & Business Logic

* **Semantic Search:** Using SentenceTransformer embeddings and FAISS indexing (`step1.py`) for efficient retrieval of relevant document chunks based on topic/query similarity (`step2.py`).
* **LLM-Based Generation:** Employing large language models (Gemma) for generating research questions and report sections based on provided context (`step2.py`, `rewrite_function.py`). Includes custom prompt engineering and response cleaning.
* **Multi-Agent Review:** Using the AutoGen framework (`step3.py`) to simulate a review process with specialized agents focusing on different quality aspects (accuracy, clarity, structure, fact-checking) and a moderator agent to synthesize feedback.
* **Iterative Refinement:** Looping through review and rewriting stages (`step3.py`, `rewrite_function.py`) until quality thresholds are met or a maximum number of iterations is reached.
* **Quality Evaluation:** Implementing quantitative and qualitative metrics (`final_evaluation.py`) using NLP techniques (spaCy dependency parsing, textstat readability), topic modeling (LDA), and external LLM calls (GPT-4) to assess generated content.
* **LLM Fine-Tuning (LoRA):** Parameter-efficient fine-tuning of a base LLM (Mistral) using PEFT/LoRA on the generated report data (`fine_tune.py`).
* **Quantization:** Using `bitsandbytes` library to load large models (Gemma) in 4-bit precision, reducing memory requirements.

## 7. Dependencies & Libraries

Based on `environment.yml`:

* **Core ML/NLP:**
  * `pytorch`: Deep learning framework.
  * `transformers`: Hugging Face library for models (Gemma, Mistral), tokenizers, pipelines.
  * `sentence-transformers`: For generating text embeddings.
  * `faiss-cpu`: Vector similarity search library.
  * `langchain-community`, `langchain-core`: Framework for building language model applications (used for vector store integration).
  * `spacy`: Advanced NLP library (used in evaluation for parsing, NER). Needs model download (e.g., `en_core_web_lg`).
  * `nltk`: NLP toolkit (dependency for other libraries).
  * `sklearn`: Machine learning library (used for LDA in evaluation).
  * `textstat`: Readability statistics calculation.
* **LLM Interaction & Agents:**
  * `autogen` / `pyautogen`: Framework for multi-agent applications.
  * `openai`: Client library for OpenAI API (GPT-4 evaluation).
  * `huggingface-hub`: For interacting with the Hugging Face Hub (model download, login).
  * `anthropic`: (Listed, but usage not apparent in analyzed scripts).
* **Efficiency & Training:**
  * `bitsandbytes`: For model quantization (loading large models with less memory).
  * `peft`: Parameter-Efficient Fine-Tuning library (for LoRA).
  * `accelerate`: Simplifies running PyTorch models across different hardware setups.
  * `datasets`: Hugging Face library for handling datasets during fine-tuning.
* **Utilities & Data Handling:**
  * `numpy`: Numerical computation.
  * `pandas`: Data manipulation (likely used internally by dependencies).
  * `requests`: For making HTTP requests (CrossRef API).
  * `python-dotenv`: For loading environment variables from `.env` files.
  * `tqdm`: Progress bars.
  * `pyyaml`: For handling YAML files (like `environment.yml`).
* **CUDA Toolkit:** `cudatoolkit=11.8` specified for GPU acceleration compatibility.

## 8. Configuration & Environment

* **Environment Variables:**
  * `HUGGINGFACE_TOKEN`: Required for downloading/accessing certain Hugging Face models. Loaded via `dotenv`.
  * `OPENAI_API_KEY`: Required for quality evaluation using GPT-4 in `final_evaluation.py`. Loaded via `dotenv`.
  * `CUDA_VISIBLE_DEVICES`: Often set in batch scripts to specify GPU usage.
  * `PYTORCH_CUDA_ALLOC_CONF`: Set in scripts (`step2.py`, `step4.py`, `rewrite_function.py`) to configure PyTorch's CUDA memory allocator (e.g., `max_split_size_mb`).
  * `CUDA_LAUNCH_BLOCKING`: Set in some batch scripts for debugging CUDA errors.
* **Configuration Files:**
  * `environment.yml`: Defines Conda environment dependencies.
  * Scripts (`step1.py`, `step2.py`, etc.) contain hardcoded parameters (model names, directories, chunk sizes) that might be externalized.
* **Execution Environment:**
  * Designed for a Linux-based HPC environment using SLURM (`batch_script*.sh`).
  * Requires Conda for environment management.
  * Relies heavily on NVIDIA GPUs (Ampere or Pascal specified in scripts) with specific CUDA versions (11.8 or 12.5).

## 9. Getting Started / Setup

Based on `README.md` and observed files:

1. **Prerequisites:** Access to an HPC cluster with SLURM, NVIDIA GPUs (Ampere recommended), Conda installed, Git.
2. **Clone Repository:** `git clone <repository_url>`
3. **Create Environment:** `conda env create -f environment.yml`
4. **Activate Environment:** `conda activate new_env` (or `llm_env` depending on the script used)
5. **API Keys:** Create a `.env` file in the project root and add:

    ```
    HUGGINGFACE_TOKEN=your_huggingface_read_token
    OPENAI_API_KEY=your_openai_api_key
    ```

6. **Prepare Input Data:** Place source markdown documents into the `./files_mmd/` directory.
7. **Configure Batch Script:** Edit `batch_script.sh` (or a variant) to:
    * Update the email address (`--mail-user`).
    * Ensure required steps are uncommented.
    * Adjust resource requests (`--gres`, `--mem`, `-t`) if necessary for the target cluster/models.
8. **Submit Job:** `sbatch batch_script.sh`
9. **Monitor:** Check Slurm output/error files (`%x_%j.out`, `%x_%j.err`) and log files in `./logs/`.
10. **Outputs:** Find results in `./embeddings`, `./initial_chapters`, `./outputs`, `./chapter_markdowns`, `./chapter_diagrams`, `./fine_tuned_model`.
