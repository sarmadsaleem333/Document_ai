import argparse
import json
from pathlib import Path

from classifier import classify_document
from extractor import load_documents
from parser import extract_bill, extract_invoice, extract_resume
from retrieval import answer_question, build_search_system, format_search_results, search_documents


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "output.json"


def build_classification_output():
    documents = load_documents(str(DATA_DIR))

    results = {}

    for filename, text in documents.items():
        print(f"Processing: {filename}")

        doc_class = classify_document(text)
        item = {"class": doc_class}

        if doc_class == "Invoice":
            item.update(extract_invoice(text))
        elif doc_class == "Resume":
            item.update(extract_resume(text))
        elif doc_class == "Utility Bill":
            item.update(extract_bill(text))

        results[filename] = item

    return results


def save_output(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def run_classification():
    results = build_classification_output()
    save_output(results)
    print("\nDone!")
    print(f"Output saved to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


def run_search(query, top_k=3):
    search_system = build_search_system(str(DATA_DIR))
    results = search_documents(query, search_system, top_k=top_k)
    print(format_search_results(results))


def run_qa(question, top_k=3, llm_model_name=None):
    search_system = build_search_system(str(DATA_DIR))
    result = answer_question(question, search_system, top_k=top_k, llm_model_name=llm_model_name)
    print(json.dumps(result, indent=4, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Document classification and semantic retrieval")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="Search documents by meaning")
    search_parser.add_argument("query", help="Semantic search query")
    search_parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")

    qa_parser = subparsers.add_parser("qa", help="Answer a question using retrieved context")
    qa_parser.add_argument("question", help="Question to answer locally")
    qa_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    qa_parser.add_argument("--llm-model", default=None, help="Optional local Hugging Face text-generation model")

    args = parser.parse_args()

    if args.command == "search":
        run_search(args.query, top_k=args.top_k)
    elif args.command == "qa":
        run_qa(args.question, top_k=args.top_k, llm_model_name=args.llm_model)
    else:
        run_classification()


if __name__ == "__main__":
    main()