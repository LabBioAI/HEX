# Baselines/HtmlRag/run_html_rag_experiments.py
import sys
import os
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, "React"))

from rag import compile_html_rag_baseline

EVALUATION_TASKS = [
    {
        "id": "hospitals",
        "task_name": "Hospitals",
        "query": "Find me all the hospitals located in Ontario."
    },
    {
        "id": "mental_health",
        "task_name": "Community Mental Health",
        "query": "Community-based mental health services in Ontario"
    },
    {
        "id": "addiction",
        "task_name": "Addiction treatment",
        "query": "Retrieve all addiction treatment centres in the Ontario."
    }
]

if __name__ == "__main__":
    print("====== INITIATING SOTA HTML-RAG UNIVERSAL DECOUPLED EXPERIMENTS RUNNER ======\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(root_dir, ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    output_dir = os.path.join(root_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    html_rag_graph = compile_html_rag_baseline()

    for task in EVALUATION_TASKS:
        print(f"\n==================================================")
        print(f"[RUNNING HTML-RAG] Task Category: {task['task_name']}")
        print(f"==================================================")

        results_state = html_rag_graph.invoke({
            "task_query": task["query"],
            "discovered_urls": [],
            "raw_markdown_corpus": "",
            "extracted_table": []
        })

        extracted_rows = results_state.get("extracted_table", [])
        total_extracted = len(extracted_rows)
        unique_names = {row["entity_name"] for row in extracted_rows if "entity_name" in row}
        total_unique = len(unique_names)

        output_payload = {
            "task_id": task["id"],
            "task_name": task["task_name"],
            "query": task["query"],
            "total_extracted_rows_raw": total_extracted,
            "total_unique_entities": total_unique,
            "extracted_data_table": extracted_rows
        }

        output_file_path = os.path.join(output_dir, f"html_rag_baseline_{task['id']}.json")
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            json.dump(output_payload, out_f, indent=2)

        print(f"\n[SUMMARY FOR HTML-RAG {task['task_name']}]:")
        print(f"  - Total Raw Records Extracted: {total_extracted}")
        print(f"  - Unique Entity Matches: {total_unique}")
        print(f"  - Output Target Path Location: {output_file_path}")

    print("\n====== ALL UNIVERSAL HTML-RAG EVALUATION BATCH RUNS COMPLETE ======")
