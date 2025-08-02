# BDD Specification: step1.py

**Date:** 2025-04-26

## Feature: Document Indexing for Semantic Search

As a user of the report generation system,
I want to process a collection of source Markdown documents
So that I can create a searchable vector index for later retrieval of relevant information.

## Key Scenarios

### Scenario: Successful Index Creation from Markdown Files

* **Given** a directory named `./files_mmd` exists and contains one or more valid Markdown files (`.mmd`).
* **And** an output directory named `./embeddings` exists or can be created by the script.
* **And** the embedding model `all-MiniLM-L6-v2` is accessible for download or is cached.
* **When** the `step1.py` script is executed.
* **Then** a FAISS index file named `faiss.index` should be created within the `./embeddings` directory.
* **And** a metadata file named `metadata.npy` should be created within the `./embeddings` directory.
* **And** a human-readable metadata file named `metadata.json` should be created within the `./embeddings` directory.
* **And** an index statistics file named `stats.json` should be created within the `./embeddings` directory.
* **And** the `metadata.json` file should contain entries corresponding to chunks from the input files, including fields like `file_name`, `title`, `authors`, `year`, `abstract`, `chunk_id`, `total_chunks`, and `excerpt`.
* **And** the `stats.json` file should contain information like the total number of documents processed, total chunks created, and the embedding model used.
* **And** the execution log (`document_indexing.log` and console output) should indicate successful processing of files and creation of the index.

### Scenario: Extracting Metadata from a Well-Formed Document

* **Given** a Markdown document content string containing:
  * An H1 heading (e.g., `# My Document Title`).
  * An author line (e.g., `By John Doe and Jane Smith`).
  * A 4-digit year (e.g., `Published 2023`).
  * An abstract section (e.g., `Abstract: This document describes...`).
* **And** the filename is `example_doc.mmd`.
* **When** the `extract_metadata` function is called with the content string and filename.
* **Then** a dictionary should be returned.
* **And** the dictionary should contain the key `title` with the value "My Document Title".
* **And** the dictionary should contain the key `authors` with the value `["John Doe", "Jane Smith"]`.
* **And** the dictionary should contain the key `year` with the value "2023".
* **And** the dictionary should contain the key `abstract` with the value "This document describes...".
* **And** the dictionary should contain the key `id` with the value `example_doc.mmd`.

### Scenario: Handling Missing Metadata in a Document

* **Given** a Markdown document content string that lacks standard H1 title, author, year, or abstract patterns.
* **And** the filename is `plain_doc.mmd`.
* **When** the `extract_metadata` function is called with the content string and filename.
* **Then** a dictionary should be returned.
* **And** the dictionary should contain the key `title` with a default value derived from the filename (e.g., "plain_doc").
* **And** the dictionary should contain the key `authors` with an empty list `[]`.
* **And** the dictionary should contain the key `year` with an empty string or "N/A".
* **And** the dictionary should contain the key `abstract` with an empty string or "N/A".
* **And** the dictionary should contain the key `id` with the value `plain_doc.mmd`.

### Scenario: Chunking a Document into Overlapping Pieces

* **Given** a document content string longer than the defined `CHUNK_SIZE` (e.g., 1000 characters).
* **And** a `CHUNK_SIZE` of 1000 and a `CHUNK_OVERLAP` of 200 are configured.
* **When** the `chunk_document` function is called with the content string, chunk size, and overlap.
* **Then** a list of strings (chunks) should be returned.
* **And** the number of chunks should be greater than 1.
* **And** the first chunk should contain approximately the first 1000 characters of the content.
* **And** the second chunk should start with roughly the last 200 characters of the first chunk.

### Scenario: Handling Errors During File Processing

* **Given** the input directory `./files_mmd` contains a mix of valid `.mmd` files and one file that is corrupted or causes a reading error.
* **When** the `step1.py` script is executed.
* **Then** an error message related to the problematic file should be logged (to `document_indexing.log` and console).
* **And** the script should continue processing the remaining valid `.mmd` files.
* **And** the final FAISS index and metadata files should be created in `./embeddings` based *only* on the content from the successfully processed files.

## Key Components Involved (Behavioral Roles)

* **`read_markdown_files`:** Finds and reads source `.mmd` documents, coordinates metadata extraction and chunking for each valid file found.
* **`extract_metadata`:** Identifies and retrieves specific details (title, authors, year, abstract) from the text content of a document based on predefined patterns. Provides default values if patterns aren't found.
* **`chunk_document`:** Breaks down a document's text into smaller, overlapping segments suitable for vector embedding.
* **`create_faiss_database`:** Takes the collection of text chunks, generates a vector representation (embedding) for each using an external model, builds a searchable vector index (FAISS), and saves the index along with detailed metadata about each chunk.
* **`main`:** Initiates the entire indexing process by sequentially calling the reading/chunking component and then the index creation component.

## Inputs and Outputs (Behavioral)

* **Input:** Path to a directory (`./files_mmd`) containing source documents in Markdown format (`.mmd`).
* **Output:**
  * A FAISS vector index file (`./embeddings/faiss.index`).
  * A NumPy file containing metadata for each indexed chunk (`./embeddings/metadata.npy`).
  * A JSON file containing the same metadata in human-readable format (`./embeddings/metadata.json`).
  * A JSON file containing statistics about the indexing process (`./embeddings/stats.json`).
  * A log file (`document_indexing.log`) detailing the execution steps and any errors encountered.

## Interactions with Other Components

* **File System:** Reads `.mmd` files from the input directory; Writes index, metadata, stats, and log files to their respective directories.
* **SentenceTransformer Library:** Loads and utilizes the `all-MiniLM-L6-v2` model to generate vector embeddings for text chunks.
* **FAISS Library:** Uses FAISS to construct and save the `IndexFlatL2` vector index.
* **NumPy Library:** Used to manage the array of vector embeddings and save/load the metadata array.
* **Logging Module:** Used to record progress, information, and errors during execution to both the console and a log file.
