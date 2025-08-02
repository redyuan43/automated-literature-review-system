# Whole-system-Masters-Project

This project implements an automated pipeline for generating, reviewing, refining, and evaluating academic literature reviews or technical reports using Large Language Models (LLMs), vector databases, and agent-based systems.

## Project Overview

The system follows a **Sequential Pipeline Architecture** designed to:

1. Create a searchable vector knowledge base from a collection of source documents
2. Generate initial reports based on research questions using relevant context
3. Implement an iterative review process with specialized AI agents
4. Rewrite reports based on consolidated feedback
5. Visualize key concepts using Mermaid diagrams
6. Evaluate the quality of generated content using multiple metrics
7. Optionally fine-tune a language model on the generated content

**Prerequisites:**

* Access to an HPC cluster with Slurm and NVIDIA GPUs (Ampere architecture recommended)
* Conda installed
* A Hugging Face token stored in an environment variable `HUGGINGFACE_TOKEN`. You can set this in your shell profile (e.g., `.bashrc`) or create a `.env` file in the project root:

    ```
    HUGGINGFACE_TOKEN=your_hugging_face_token_here
    OPENAI_API_KEY=your_openai_api_key (for evaluation functions)
    ```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ruipcarvalho/Isla_MEng_2025_latest.git
cd Isla_MEng_2025_latest
```

### 2. Create the Conda Environment

Create the required conda environment using the provided `environment.yml` file. This might take some time as it installs numerous packages including PyTorch with CUDA support.

```bash
conda env create -f environment.yml
```

This will create an environment named `new_env` (as specified in the file).

### 3. Activate the Environment

Before running any scripts or submitting the job, activate the environment:

```bash
conda activate new_env
```

## Running the Workflow (HPC Cluster)

The primary workflow is designed to run as a batch job on an HPC cluster using Slurm.

### Submit the Batch Job

The `batch_script.sh` script handles loading necessary modules (CUDA/cuDNN), activating the conda environment, and running the Python scripts (`step1.py`, `step2.py`, etc.) in sequence.

Make sure the email address in `batch_script.sh` is updated to your own if you want email notifications:

```bash
#SBATCH --mail-user=your_email@example.com
```

Submit the job to the Slurm scheduler:

```bash
sbatch batch_script.sh
```

### Monitoring the Job

You can monitor the job's status using Slurm commands:

* `squeue -u your_username`: Shows your running or pending jobs
* `scontrol show job job_id`: Shows detailed information about a specific job

Output and error logs will be written to files named `Isla_job_id.out` and `Isla_job_id.err` respectively (where `job_id` is the Slurm job ID).

## Pipeline Architecture

The project follows a modular, step-by-step approach with file-based communication between stages:

### Step 1: Document Indexing (`step1.py`)

* Processes source documents (Markdown) from `./files_mmd/`

* Extracts metadata (title, authors, year, abstract)
* Chunks content with overlap to maintain context
* Generates embeddings using SentenceTransformer
* Creates a searchable FAISS index and saves it to `./embeddings/`

### Step 2: Initial Report Generation (`step2.py`)

* Generates research questions based on a central topic (e.g., "semiconductors")

* For each question:
  * Queries the FAISS index for relevant context
  * Uses Gemma LLM (27B parameters) to generate structured reports
  * Includes sections: Background Knowledge, Current Research, Recommendations
  * Formats references with CrossRef API when possible
  * Checks for content repetition between generated chapters
* Saves the initial reports to `./initial_chapters/` in JSON format

### Step 3: Iterative Content Improvement (`step3.py`)

* Implements a multi-agent review system using AutoGen

* Specialized agents focus on:
  * Technical Accuracy
  * Clarity
  * Structure
  * Fact-Checking
* A Moderator agent synthesizes feedback into improvement points
* Reports are iteratively rewritten until quality thresholds are met
* Final improved chapters are saved to `./chapter_markdowns/` in Markdown format

### Step 4: Concept Diagram Generation (`step4.py`)

* Takes the final markdown chapters

* Uses Gemma LLM to generate Mermaid concept diagrams
* Saves diagrams to `./chapter_diagrams/`

### Step 5: Model Fine-Tuning (`fine_tune.py`) (Optional)

* Uses the final markdown chapters as training data

* Fine-tunes a Mistral-7B model using PEFT/LoRA
* Saves the fine-tuned model to `./fine_tuned_model/`

## Report Evaluation Framework

The evaluation framework in `final_evaluation.py` assesses generated reports based on:

### Technical Depth (45%)

* **Technical Term Count:** Density of domain-specific terminology
* **Concept Hierarchy Depth:** Using topic modeling with LDA
* **Syntactic Complexity:** Based on dependency parsing
* **LLM Assessment:** GPT-4 evaluation of technical accuracy and depth

### Clarity & Understandability (35%)

* **Flesch Reading Ease Score:** Adjusted for technical content
* **Defined Terms Count:** Explanations of technical concepts
* **Example Count:** Practical illustrations of concepts
* **Contextual Coherence:** Semantic flow between paragraphs
* **LLM Assessment:** GPT-4 evaluation of clarity and accessibility

### Structure (20%)

* **Topic Modeling:** Using LDA to measure organization
* **Local Coherence:** Flow between adjacent paragraphs
* **Thematic Consistency:** Overall logical progression
* **LLM Assessment:** GPT-4 evaluation of document structure

### Citation Accuracy

* **Claim Verification:** Matching claims against referenced paper content
* **Citation Completeness:** Identifying missing or incorrect citations

![Report Evaluation Metrics Visualization](https://github.com/user-attachments/assets/b1ad2aa7-c1cc-4a69-829b-4ccde9c74a92)
![Report Evaluation Categories](https://github.com/user-attachments/assets/ff800367-c48c-41e6-9cbf-692e40ae1ecf)

## Key Technologies

* **LLM Models:**
  * Gemma-3-27B (Initial generation, rewriting, diagram creation)
  * Mistral-7B (For fine-tuning)
  * GPT-4 (For evaluation, if API key available)
* **Embeddings & Search:**
  * SentenceTransformer (all-MiniLM-L6-v2)
  * FAISS for vector similarity search
  * LangChain for vector store integration
* **Agent Framework:**
  * AutoGen for multi-agent orchestration
* **Efficiency & Training:**
  * BitsAndBytes for 4-bit quantization
  * PEFT/LoRA for parameter-efficient fine-tuning
* **NLP & Evaluation:**
  * spaCy for linguistic analysis
  * TextStat for readability metrics
  * Scikit-learn for topic modeling (LDA)

## Repository Structure

```
Whole-system-Masters-Project/
├── batch_script.sh              # Main SLURM job script
├── environment.yml              # Conda environment definition
├── step1.py                     # Document indexing
├── step2.py                     # Initial report generation
├── step3.py                     # Multi-agent review & improvement
├── step4.py                     # Concept diagram generation
├── rewrite_function.py          # Text rewriting utilities
├── final_evaluation.py          # Quality assessment functions
├── fine_tune.py                 # LLM fine-tuning (optional)
├── files_mmd/                   # Input markdown documents
├── embeddings/                  # FAISS index & metadata
├── initial_chapters/            # Generated JSON reports
├── outputs/                     # Review outputs
├── chapter_markdowns/           # Final improved reports
├── chapter_diagrams/            # Generated Mermaid diagrams
├── logs/                        # Log files
└── cursor/                      # Detailed documentation
```

## System Diagram (Example)

```mermaid
graph TD
    A[EV Demand & WBG Adoption] --> B(SiC & GaN Materials);
    B --> C{Material Science Challenges};
    C --> D[Defect Engineering];
    D --> E[Dislocation Formation (Deep Trenches)];
    D --> F[Impurity Control & Carrier Lifetime];
    C --> G[Dopant Control];
    C --> H[Gate Driver Integration];
    H --> I[High-Temp Stability & Low Inductance];
    D --> J[Gate Oxide Reliability];
    A --> K[Scalable Production & Cost Reduction];
    K --> L{Research Focus};
    L --> M[Novel Gettering & Annealing];
    L --> N[Impurity Management];
    L --> O[Advanced Gate Driver Tech (High-k Dielectrics)];
    O --> P[Improved Device Performance];
    P --> K;
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#ccf,stroke:#333,stroke-width:2px
```

## Data Flow Overview

1. Input: Raw documents (`.mmd` files) in `./files_mmd/`
2. `step1.py`: Creates FAISS index and metadata in `./embeddings/`
3. `step2.py`: Generates JSON reports in `./initial_chapters/`
4. `step3.py`: Iteratively improves reports, saving output to:
   * `./outputs/` (JSON feedback and quality assessments)
   * `./chapter_markdowns/` (Final markdown reports)
5. `step4.py`: Generates concept diagrams in `./chapter_diagrams/`
6. `fine_tune.py`: Fine-tunes a model using `./chapter_markdowns/`, saving to `./fine_tuned_model/`

For more detailed documentation, see the files in the `./cursor/` directory.
