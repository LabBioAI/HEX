# Baselines/React/run_experiments.py

import sys
import os

# Force Python to find modules in the same directory as this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Your existing imports continue below...
# Baselines/React/run_experiments.py
import sys
import os
import json  

# Force Python to find modules in the same directory as this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from react_baseline import compile_react_baseline

# Definition of your paper's core macro-region evaluation dataset corpus
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
        "query": "Retrieve all addiction treatment programs in the Ontario."
    }
]

if __name__ == "__main__":
    print("====== INITIATING SOTA REACT AGGREGATED EXPERIMENTS RUNNER ======\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    env_path = os.path.join(root_dir, ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    output_dir = os.path.join(root_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    react_graph = compile_react_baseline()

    for task in EVALUATION_TASKS:
        print(f"\n==================================================")
        print(f"[RUNNING] Task Category: {task['task_name']}")
        print(f"==================================================")

        results_state = react_graph.invoke({
            "task_query": task["query"],
            "trajectory": [],
            "loop_count": 0,
            "extracted_table": []
        })

        aggregated_rows = results_state.get("extracted_table", [])
        total_extracted = len(aggregated_rows)
        unique_names = {row["entity_name"] for row in aggregated_rows if "entity_name" in row}
        total_unique = len(unique_names)

        output_payload = {
            "task_id": task["id"],
            "task_name": task["task_name"],
            "query": task["query"],
            "total_extracted_rows_raw": total_extracted,
            "total_unique_entities": total_unique,
            "extracted_data_table": aggregated_rows
        }

        output_file_path = os.path.join(output_dir, f"react_baseline_{task['id']}.json")
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            json.dump(output_payload, out_f, indent=2)

        print(f"\n[SUMMARY FOR {task['task_name']}]:")
        print(f"  - Total Raw Records Extracted: {total_extracted}")
        print(f"  - Unique Entity Matches: {total_unique}")
        print(f"  - Structured Dataset Table Persistent Logged to: {output_file_path}")

    print("\n====== ALL BATCH PROCESS EVALUATION TASKS COMPLETED ======")
