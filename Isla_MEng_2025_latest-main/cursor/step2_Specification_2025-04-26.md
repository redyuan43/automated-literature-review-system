# Technical Specification: step2.py

**Date:** 2025-04-26

## 1. File Overview

* **Purpose:** This script forms the second major step in the project's pipeline. Its primary function is to generate initial structured academic reports based on specific topics or research questions. It achieves this by retrieving relevant context from a pre-existing vector knowledge base (created by `step1.py`), using this context to prompt a large language model (Gemma 27B), and structuring the generated content into report sections (e.g., Background, Current Research, Recommendations, References). It also includes functionality for generating research questions, enhancing citation information via the CrossRef API, and attempting to mitigate repetitive content generation.
* **Role in Project:** Consumes the vector index from `step1.py` and produces initial report drafts (as JSON files) that serve as input for the review and refinement process in `step3.py`.

## 2. Key Classes

* **`CustomGemmaClient`**
  * **Responsibility:** Provides an interface compatible with the Autogen framework for interacting with a locally loaded, quantized Gemma LLM.
  * **Key Attributes:** `config` (model/client configuration), `model_name`, `device`, `gen_params` (generation parameters), `model`, `tokenizer`.
  * **Key Methods:**
    * `__init__`: Initializes the client, loads the shared model using `load_shared_model`.
    * `_format_chat_prompt`: Formats a list of messages into the specific chat template required by Gemma models.
    * `create`: Takes Autogen-style message parameters, formats the prompt, generates text using the loaded Gemma model (with error handling/fallback to greedy decoding), cleans the output, and returns it in an Autogen-compatible format.
    * `message_retrieval`: Extracts the content from the response object.
    * `cost`, `get_usage`: Placeholder methods for compatibility (return 0 or empty dict).
  * **Core Functionality:** Acts as a bridge between the Autogen framework (used later in `step3.py`, though the client might be defined here for consistency or early use) and the underlying Hugging Face `transformers` model, handling Gemma-specific prompt formatting and generation parameters.
* **`ContentEmbeddingStore`**
  * **Responsibility:** Manages a simple in-memory vector store (FAISS) of text content (summaries and full text of generated sections) to facilitate similarity checks.
  * **Key Attributes:** `embedding_model` (SentenceTransformer), `content_store` (FAISS index).
  * **Key Methods:**
    * `__init__`: Initializes the embedding model and the FAISS store (with a temporary placeholder).
    * `_remove_placeholder`: Removes the initial placeholder from the FAISS index.
    * `add_content`: Takes text content, generates a summary (using `summarize_section`), embeds both full text and summary, and adds them to the FAISS store with metadata.
    * `check_similarity`: Takes new content, generates its summary, performs similarity searches against the store for both full text and summary, and returns whether a similarity above a threshold is found, along with the metadata of the similar item.
  * **Core Functionality:** Provides a mechanism to detect if newly generated content is too similar to content generated previously within the same execution run, primarily used by `check_section_repetition`.

## 3. Key Functions

* **`load_shared_model(model_name, device)`**: Loads a Hugging Face model (specifically Gemma 27B) and tokenizer with specified quantization (4-bit) and memory settings. Caches the loaded model/tokenizer in the `_model_cache` dictionary to prevent reloading. Handles GPU memory management and error logging during loading. (Note: Similar logic likely exists in `rewrite_function.py`).
* **`load_vector_store()`**: Loads the FAISS index and associated metadata created by `step1.py` from the `./embeddings` directory. It handles potential conversion/copying to a `./converted_index` directory to match LangChain's expected format (`index.faiss`). It creates an `InMemoryDocstore` from the metadata and returns a LangChain `FAISS` vector store object. Includes error handling and size mismatch checks.
* **`get_paper_context(topic, num_papers=8)`**: Queries the loaded vector store using the input `topic` to find relevant document chunks. Groups chunks by paper title, calculates average relevance scores, selects the top `num_papers`, and formats the combined content of these papers into a context string suitable for an LLM prompt. Returns the context string and a list of dictionaries containing metadata for the selected papers.
* **`get_crossref_citation(title: str) -> Optional[Dict]`**: Takes a paper title, queries the CrossRef API (`api.crossref.org`) via HTTP GET request, attempts to find the best match based on title similarity, and returns a dictionary with structured citation information (DOI, authors, journal, year, etc.) if found.
* **`clean_metadata(metadata)`**: Takes a paper metadata dictionary and attempts to clean/validate the title, authors, and year fields using regex and predefined validation logic (`validate_author`).
* **`organize_papers_for_citation(paper_metadata)`**: Processes the list of paper metadata dictionaries retrieved from the vector store, cleans the metadata using `clean_metadata`, assigns sequential citation IDs (`[1]`, `[2]`, etc.), and returns a dictionary mapping cleaned titles to paper details (including citation ID) and a separate map of title to citation ID.
* **`format_reference_section(organized_papers)`**: Generates a list of formatted reference strings based on the `organized_papers` dictionary. It prioritizes using data fetched from CrossRef (if available via `get_crossref_citation` and stored within the `organized_papers` structure) for a standard citation style, otherwise creates a basic reference from available metadata.
* **`call_model(prompt, system_message=None, temperature=0.7, max_tokens=1000)`**: A wrapper function to invoke the `CustomGemmaClient`. It takes a user prompt and optional system message, instantiates the client, sends the request, retrieves the response, and applies basic cleaning (`clean_response`). Includes logic to handle potentially insufficient responses.
* **`generate_report(topic, max_retries=2, num_papers=4, embedding_store=None, temperature=0.7)`**: The main function for generating a single chapter/report. It orchestrates calls to `get_paper_context`, `organize_papers_for_citation`, `get_crossref_citation` (optional), `call_model` (for each section: Background, Current Research, Recommendations), `format_reference_section`, and potentially `check_section_repetition`. Includes a retry loop for robustness. Returns the final report string, a dictionary of sections, and the organized paper metadata.
* **`generate_research_questions(topic: str, num_questions: int = 3)`**: Uses the LLM (`call_model`) with specific prompts to generate a list of diverse research questions related to the input `topic`. Implements batching and checks for duplicates to ensure variety. Includes fallback questions if generation fails.
* **`summarize_section(content, max_length=150)`**: A simple utility function that calls the LLM (`call_model`) to generate a concise summary of the input `content`. Used by `ContentEmbeddingStore`.
* **`check_section_repetition(report_sections, organized_papers, question, embedding_store)`**: Iterates through generated `report_sections`, uses the `embedding_store` (`ContentEmbeddingStore`) instance to check for similarity against previously added content. If similarity exceeds a threshold, it triggers `rewrite_section` and updates the section content. Returns the potentially modified `report_sections` dictionary.
* **`rewrite_section(section_name, current_content, similar_content, question, organized_papers)`**: Constructs a specific prompt instructing the LLM (`call_model`) to rewrite the `current_content` to be distinct from the `similar_content`, using a different perspective while referencing the provided `organized_papers`.
* **`main()`**: Script entry point. Sets a main `topic`, calls `generate_research_questions`, then loops through each question, calling `generate_report`. It formats the output of `generate_report` into a final JSON structure (`report_data`) and saves it to `./initial_chapters/chapter_{chapter_num}.json`. Manages GPU memory between chapter generations using `clear_memory` and `monitor_gpu_memory`.
* **Utility Functions:** `clear_memory`, `monitor_gpu_memory`, `clean_response` (removes model artifacts), `extract_sections` (splits report text by `##` headers), `extract_authors` (regex-based author extraction attempt), `validate_author` (checks if a string looks like a valid author name).

## 4. Data Structures / Constants

* **Environment Variables (Read):** `HUGGINGFACE_TOKEN`, `CUDA_VISIBLE_DEVICES`.
* **Environment Variables (Set):** `OAI_CONFIG_LIST` (JSON string defining Gemma model config for Autogen client), `PYTORCH_CUDA_ALLOC_CONF`, `CUDA_LAUNCH_BLOCKING`.
* **Key Data Structures:**
  * `paper_metadata` (List[Dict]): List of dictionaries, each representing a retrieved paper chunk/document with fields like `title`, `authors`, `year`, `abstract`, `chunks`, `chunk_scores`, `id`. Generated by `get_paper_context`.
  * `organized_papers` (Dict[str, Dict]): Dictionary mapping cleaned paper titles to their metadata, including assigned `citation_id` and potentially `crossref_citation` data. Generated by `organize_papers_for_citation`.
  * `report_data` (Dict): The final JSON structure saved for each chapter. Contains `question`, `domain`, `timestamp`, `sections` (dict mapping section names to content), `referenced_papers` (the `organized_papers` dict), and generation `metadata`.
  * `_model_cache` (Dict): Global dictionary used by `load_shared_model` to cache loaded models and tokenizers.

## 5. Logic Flow & Control

* The script is intended to be run directly (`if __name__ == "__main__":`).
* Execution starts in `main()`.
* Initial setup includes logging, GPU memory configuration, loading the Hugging Face token, setting the `OAI_CONFIG_LIST` environment variable, and initializing the `ContentEmbeddingStore`.
* A hardcoded main `topic` ("semiconductors") is used.
* `generate_research_questions` is called to produce a list of questions.
* The script then enters a loop iterating through the generated `questions`.
* Inside the loop for each `question`:
  * GPU memory is monitored/cleared.
  * `generate_report` is called with the current `question`.
    * This involves vector store loading, similarity search, context formatting, multiple LLM calls for section generation, potential CrossRef lookups, reference formatting, and potentially similarity checking/rewriting via `check_section_repetition` and `rewrite_section`.
  * The results from `generate_report` are packaged into the `report_data` dictionary.
  * `report_data` is saved as a JSON file (e.g., `initial_chapters/chapter_1.json`).
  * GPU memory is cleared again before the next iteration.
* Error handling (try-except blocks) is present in several key functions like `load_shared_model`, `load_vector_store`, `get_paper_context`, `call_model`, `generate_report`, and the main loop in `main`. Retries are implemented within `generate_report`.

## 6. External Interactions

* **Imports:** `time`, `types`, `transformers`, `torch`, `json`, `os`, `autogen`, `gc`, `langchain_community`, `tempfile`, `huggingface_hub`, `dotenv`, `requests`, `re`, `numpy`, `logging`, `sys`, `datetime`, `functools`, `pickle`, `shutil`, `faiss`.
* **File System Reads:**
  * `.env` file (via `dotenv`).
  * FAISS index (`faiss.index`) and metadata (`metadata.npy`) from `./embeddings` (or `./converted_index`).
* **File System Writes:**
  * Chapter output JSON files to `./initial_chapters/`.
  * Potentially creates `./converted_index/` and copies `index.faiss` into it.
  * Potentially creates `./offload/` for model layer offloading.
  * Writes log messages (configured via `logging`).
* **External Libraries Called:**
  * `transformers`: Loading models/tokenizers, generation (`model.generate`), quantization (`BitsAndBytesConfig`).
  * `torch`: Tensor operations, GPU management (`cuda.empty_cache`, memory stats).
  * `langchain_community`: Loading FAISS store (`FAISS`, `InMemoryDocstore`), embeddings (`HuggingFaceEmbeddings`).
  * `faiss`: Reading the raw index (`faiss.read_index`).
  * `huggingface_hub`: Logging in (`login`).
  * `requests`: Making HTTP calls to CrossRef API.
  * `autogen`: Used implicitly via setting `OAI_CONFIG_LIST` for `CustomGemmaClient` compatibility.
  * `numpy`: Array operations, loading metadata (`np.load`).
  * `dotenv`: Loading `.env` file.
  * `re`: Regular expressions for cleaning and validation.
* **Exports/Intended Use by Others:**
  * Primary output is the set of JSON files in `./initial_chapters`, intended for consumption by `step3.py`.
  * The `CustomGemmaClient` and potentially `load_shared_model` might be considered reusable components, although similar logic appears in `rewrite_function.py`.

## 7. Assumptions / Dependencies

* **Prior Steps:** Assumes `step1.py` has executed successfully and its output (`./embeddings/faiss.index`, `./embeddings/metadata.npy`) exists and is valid.
* **Environment:** Assumes a Python environment matching `environment.yml` is active. Requires specific libraries like `transformers`, `torch` (with CUDA), `langchain`, `faiss`, `autogen`, `requests`, `dotenv`.
* **Hardware:** Critically depends on having a compatible NVIDIA GPU with sufficient VRAM (~70GiB allocated in settings) and CUDA installed (likely 11.8 based on `environment.yml`) to run the Gemma 27B 4-bit model. CPU fallback is mentioned but likely impractical for the 27B model.
* **API Keys/Tokens:** Requires `HUGGINGFACE_TOKEN` to be set in the environment or `.env` file for model access.
* **Network:** Requires internet access for Hugging Face Hub downloads and CrossRef API calls.
* **Configuration:** Relies on environment variables (`HUGGINGFACE_TOKEN`, `CUDA_VISIBLE_DEVICES`) and internal constants/configurations (like the hardcoded `topic` and the structure set in `OAI_CONFIG_LIST`).
* **File Permissions:** Requires write access to `./initial_chapters`, `./converted_index` (if created), `./offload` (if created), and the logging directory.
