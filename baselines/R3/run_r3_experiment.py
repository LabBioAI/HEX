# Baselines/R3/run_r3_experiments.py
import sys
import os
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mount workspace paths pointing back to your shared components
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, "React"))

from r3_baseline import compile_r3_baseline

# Target paper evaluation corpus matrices
EVALUATION_TASKS = [
    {
        "id": "hospitals",
        "task_name": "Hospitals",
        "query": "Find me all the hospitals located in Ontario."
    },
    {
        "id": "mental_health",
        "task_name": "Community Mental Health",
        "query": "Community-based mental health centres in Ontario"
    },
    {
        "id": "addiction",
        "task_name": "Addiction treatment",
        "query": "Retrieve all addiction treatment centres in the Ontario."
    }
]

if __name__ == "__main__":
    print("====== INITIATING SOTA REWRITE-RETRIEVE-READ (R3) EXPERIMENTS RUNNER ======\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(root_dir, ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    output_dir = os.path.join(root_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    r3_graph = compile_r3_baseline()

    for task in EVALUATION_TASKS:
        print(f"\n==================================================")
        print(f"[RUNNING R3] Task Category: {task['task_name']}")
        print(f"==================================================")

        results_state = r3_graph.invoke({
            "task_query": task["query"],
            "rewritten_query": "",
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
            "rewritten_search_query": results_state.get("rewritten_query", ""),
            "total_extracted_rows_raw": total_extracted,
            "total_unique_entities": total_unique,
            "extracted_data_table": extracted_rows
        }

        output_file_path = os.path.join(output_dir, f"r3_baseline_{task['id']}.json")
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            json.dump(output_payload, out_f, indent=2)

        print(f"\n[SUMMARY FOR R3 {task['task_name']}]:")
        print(f"  - Total Raw Records Extracted: {total_extracted}")
        print(f"  - Unique Entity Matches: {total_unique}")
        print(f"  - Output Target Path Location: {output_file_path}")

    print("\n====== ALL R3 BASELINE EVALUATION CORPUS CHECKS COMPLETE ======")
