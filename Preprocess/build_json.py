"""
This module processes markdown files from finance and insurance directories,
as well as FAQ data from a JSON file, and compiles them into separate JSON
documents suitable for further processing or analysis.

Functions:
    build_finance_json(): Processes finance markdown files and creates a JSON document.
    build_insurance_json(): Processes insurance markdown files and creates a JSON document.
    build_faq_json(): Processes FAQ data and creates a JSON document.
"""

import json
import os
import re
from tqdm import tqdm

def build_finance_json():
    """
    Processes markdown files in the finance_markdown directory and compiles them into a JSON file.

    This function performs the following steps:
        1. Iterates through each markdown file in the specified finance_markdown directory.
        2. Extracts the source from the filename.
        3. Splits the file content by the delimiter "[sep]" to separate the header and contents.
        4. Further splits the contents into chunks based on double newlines.
        5. Identifies tables by detecting the presence of "|" and "-" in a chunk.
        6. Appends each text and table chunk to the documents list with appropriate metadata.
        7. Saves the compiled documents to a JSON file at ../documents/finance.json.

    Raises:
        FileNotFoundError: If the finance_markdown directory does not exist.
        IOError: If there are issues reading a markdown file or writing the JSON output.
    """
    documents = []
    finance_markdown_path = "../finance_markdown"

    for file in tqdm(
        os.listdir(finance_markdown_path), desc="Processing finance markdown files"
    ):
        file_path = os.path.join(finance_markdown_path, file)
        source = file.split(".")[0].split("_")[0]
        with open(file_path, "r", encoding="utf8") as f:
            texts = f.read()
        texts = texts.split("[sep]")
        head = texts[0].strip()
        contents = [content.strip() for content in texts[1:]]
        tables = []
        texts = []
        for content in contents:
            chunks = content.split("\n\n")
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:
                    if "|" in chunk and "-" in chunk:
                        tables.append(chunk)
                    else:
                        texts.append(chunk)
        for text in texts:
            documents.append(
                {
                    "text": head + "\n" + text,
                    "metadata": {"source": source, "category": "finance"},
                }
            )
        for table in tables:
            documents.append(
                {
                    "text": head + "\n" + table,
                    "metadata": {"source": source, "category": "finance"},
                }
            )
    print("Number of documents in finance:", len(documents))
    output_path = "../documents/finance.json"
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

def build_insurance_json():
    """
    Processes markdown files in the insurance_markdown directory and compiles them into a JSON file.

    This function performs the following steps:
        1. Iterates through each folder in the insurance_markdown directory.
        2. For each markdown file in the folder, reads and cleans the content by:
            - Removing titles.
            - Removing images.
            - Compressing multiple newlines into double newlines.
        3. Splits the cleaned text into paragraphs based on double newlines.
        4. Appends each paragraph to the documents list with appropriate metadata.
        5. Saves the compiled documents to a JSON file at ../documents/insurance.json.

    Raises:
        FileNotFoundError: If the insurance_markdown directory does not exist.
        IOError: If there are issues reading a markdown file or writing the JSON output.
    """
    documents = []
    insurance_markdown_path = "../insurance_markdown"

    for folder in tqdm(
        os.listdir(insurance_markdown_path), desc="Processing insurance folders"
    ):
        folder_path = os.path.join(insurance_markdown_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".md"):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        texts = f.read()
                    # Remove titles
                    texts = re.sub(r"^#+.*$", "", texts, flags=re.MULTILINE)
                    # Remove images
                    texts = re.sub(r"!\[.*?\]\(.*?\)", "", texts)
                    # Compress multiple newlines
                    texts = re.sub(r"\n{2,}", "\n\n", texts)
                    texts = texts.strip()
                    for text in texts.split("\n\n"):
                        documents.append(
                            {
                                "text": text,
                                "metadata": {
                                    "source": folder,
                                    "category": "insurance",
                                },
                            }
                        )
    print("Number of documents in insurance:", len(documents))
    output_path = "../documents/insurance.json"
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

def build_faq_json():
    """
    Processes FAQ data from a JSON file and compiles it into a separate JSON document.

    This function performs the following steps:
        1. Loads FAQ data from ../pid_map_content.json.
        2. Iterates through each source and its associated Q&A lists.
        3. For each Q&A pair, combines the question and answers into a single text block.
        4. Appends each combined text to the documents list with appropriate metadata.
        5. Saves the compiled documents to a JSON file at ../documents/faq.json.

    Raises:
        FileNotFoundError: If the pid_map_content.json file does not exist.
        IOError: If there are issues reading the JSON file or writing the output.
        KeyError: If expected keys are missing in the FAQ data.
    """
    with open("../pid_map_content.json", "rb") as f:
        datas = json.load(f)
    documents = []
    for source, qa_lists in tqdm(datas.items(), desc="Processing FAQ data"):
        for qa in qa_lists:
            documents.append(
                {
                    "text": qa["question"] + "\n" + "\n".join(qa["answers"]),
                    "metadata": {
                        "source": source,
                        "category": "faq",
                    },
                }
            )
    print("Number of documents in faq:", len(documents))
    output_path = "../documents/faq.json"
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    """
    Main execution block.

    This block ensures that the ../documents directory exists and then calls the functions
    to build JSON documents for finance, insurance, and FAQ data.
    """
    documents_dir = "../documents"
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)

    build_finance_json()
    build_insurance_json()
    build_faq_json()