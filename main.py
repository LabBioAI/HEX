import yaml
import asyncio
import json
import os
import sys
from dotenv import load_dotenv
import hashlib
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add
import shutil
import csv

# Dynamically append the parent directory (Hex/) to Python's module search path
# This allows scripts inside 'main/' to see folders like 'prompts/' or 'configs/'
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Core LangChain & LangGraph Framework
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

# Search tools & Parsers
from ddgs import DDGS
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from docling.document_converter import DocumentConverter

# Import schema schema.py
from prompts.schema import EntityCollection


# 1. Loading LLM API key from .env
external_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))

if os.path.exists(external_env_path):
    load_dotenv(dotenv_path=external_env_path, override=True)
    print("Success: Local .env file detected inside Hex/ and variables initialized.")
else:
    print(f"CRITICAL ERROR: .env file not found at: {external_env_path}")
    sys.exit(1)

openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    print("CRITICAL ERROR: OPENAI_API_KEY is missing or empty inside your .env file.")
    sys.exit(1)


# Define Complete System State
class HEXState(TypedDict):
    user_query: str
    region_type: str
    city_names: List[str]
    search_queries: List[str]
    markdown_pages: List[Dict[str, str]]
    extracted_entities: Annotated[List[Dict[str, Any]], add]


# File System Helper Utilities
def load_file(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, "w") as f:
            f.write("Hospitals located in Toronto")
        return "Hospitals located in Toronto"
    with open(path, "r") as f:
        return f.read().strip()


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "model_settings": {"model_name": "gpt-4o-mini", "temperature": 0.2},
            "node2_settings": {"limit": 5},
            "ddgs_settings": {"max_results": 3},
            "node4_storage": {"memory_dir": "./memory"}
        }
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Node 1: Task Analysis
def task_analysis_node(state: HEXState) -> Dict[str, Any]:
    config = load_yaml("configs/config.yaml")
    prompt_text = load_file("prompts/analysis_prompt.txt")

    brain = ChatOpenAI(
        model=config['model_settings']['model_name'],
        temperature=config['model_settings']['temperature'],
        api_key=openai_api_key
    )

    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | brain

    print("\n" + "=" * 40)
    print("HEX STATE 1: TASK ANALYSIS")
    print(f"Current Input: {state['user_query']}")
    print("-" * 40)

    response = chain.invoke({"user_query": state["user_query"]})
    decision = response.content.lower().strip()

    print(f"Logic Result: Query is {decision.upper()}")
    print("=" * 40 + "\n")

    return {"region_type": decision}


# Node 2: Task Decomposition
def node_2_dbpedia(state: HEXState) -> Dict[str, Any]:
    config = load_yaml("configs/config.yaml")
    settings = config.get('node2_settings', {})
    limit = settings.get('limit', 50)

    ref_prompt_text = load_file("prompts/node2_prompt.txt")
    brain = ChatOpenAI(
        model=config['model_settings']['model_name'],
        temperature=0,
        api_key=openai_api_key
    )
    ref_chain = PromptTemplate.from_template(ref_prompt_text) | brain
    region_name = ref_chain.invoke({"user_query": state["user_query"]}).content.strip()

    region = f"dbr:{region_name.replace(' ', '_').replace('.', '')}"

    print("=" * 40)
    print("HEX STATE 2: KNOWLEDGE GRAPH Invoked")
    print(f"Region: {region} | Limit: {limit}")
    print("-" * 40)

    query = f"""
        PREFIX dbo: <http://dbpedia.org>
        PREFIX dbr: <http://dbpedia.org>
        PREFIX rdfs: <http://w3.org>

        SELECT DISTINCT ?name WHERE {{
          {{
            ?place dbo:subdivision {region} ;
                   a ?type ;
                   rdfs:label ?name .
            FILTER(?type IN (dbo:City, dbo:Town, dbo:Village, dbo:PopulatedPlace))
            FILTER(LANG(?name) = "en")
          }}
          UNION
          {{
            ?place dbo:isPartOf {region} ;
                   a ?type ;
                   rdfs:label ?name .
            FILTER(?type IN (dbo:City, dbo:Town, dbo:Village, dbo:PopulatedPlace))
            FILTER(LANG(?name) = "en")
          }}
        }}
        LIMIT {limit}
        """

    sparql = SPARQLWrapper("https://dbpedia.org")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod(POST)
    sparql.addCustomHttpHeader("User-Agent", "Mozilla/5.0")
    sparql.addCustomHttpHeader("Accept", "application/sparql-results+json")

    try:
        response = sparql.query()
        results = response.convert()
        bindings = results["results"]["bindings"]

        exclude = ["School", "College", "University", "Hospital", "Church", "Club", "Park", "Station"]
        raw_names = [row["name"]["value"] for row in bindings if "name" in row]

        filtered_cities = []
        for name in raw_names:
            city_clean = name.split(",")[0].strip()
            if not any(word in city_clean for word in exclude):
                filtered_cities.append(city_clean)

        final_list = sorted(list(set(filtered_cities)))
        print(f"Knowledge Graph retrieved {len(final_list)} Unique Cities.")
        return {"city_names": final_list}

    except Exception as e:
        print(f"Connection Error: {e}")
        return {"city_names": []}


# Node 3: Query Reformulation
def node_3_query_reformulation(state: HEXState) -> Dict[str, Any]:
    config = load_yaml("configs/config.yaml")
    prompt_text = load_file("prompts/node3_prompt.txt")

    brain = ChatOpenAI(
        model=config['model_settings']['model_name'],
        temperature=config['model_settings']['temperature'],
        api_key=openai_api_key
    )

    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | brain
    reformulated_queries = []

    print("=" * 40)
    print("HEX STATE 3: QUERY SYNTHESIS")
    print(f"Reformulating queries for {len(state['city_names'])} cities...")
    print("-" * 40)

    for city in state["city_names"]:
        response = chain.invoke({
            "user_query": state["user_query"],
            "city": city
        })
        query = response.content.strip()
        reformulated_queries.append(query)

    print(f"Generated {len(reformulated_queries)} specialized queries.")
    return {"search_queries": reformulated_queries}


# Node 4: Agentic Retrieval
async def node_4_ddgs_retrieval(state: HEXState) -> Dict[str, Any]:
    config = load_yaml("configs/config.yaml")
    ddgs_cfg = config.get("ddgs_settings", {"max_results": 3})
    storage_cfg = config.get("node4_storage", {"memory_dir": "./memory"})

    task_id = hashlib.md5(state["user_query"].encode()).hexdigest()
    memory_path = os.path.join(storage_cfg['memory_dir'], f"ddgs_mem_{task_id}.json")
    os.makedirs(storage_cfg['memory_dir'], exist_ok=True)

    url_memory = set()
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            url_memory = set(json.load(f))

    doc_converter = DocumentConverter()
    input_q = asyncio.Queue()
    markdown_output = []

    # Execute queries when state bypass invoked
    target_queries = state.get("search_queries", [])
    if not target_queries or len(target_queries) == 0:
        target_queries = [state["user_query"]]

    for q in target_queries:
        await input_q.put(q)

    print("=" * 40)
    print("HEX STATE 4: DUX DISTRIBUTED GLOBAL SEARCH")
    print(f"Status: Processing {len(target_queries)} Queries via DDGS")
    print("-" * 40)

    def run_ddgs_sync(query_str, max_results):
        with DDGS() as ddgs:
            return list(ddgs.text(query_str, max_results=max_results))

    async def worker_loop():
        while not input_q.empty():
            current_q = await input_q.get()
            try:
                print(f"DDGS-Search -> {current_q}")
                results = await asyncio.to_thread(run_ddgs_sync, current_q, ddgs_cfg.get("max_results", 3))

                if not results:
                    continue

                for entry in results:
                    url = entry.get("href")
                    if not url or url in url_memory:
                        continue

                    try:
                        print(f"DDGS-Docling -> Parsing: {url}")
                        render = await asyncio.to_thread(doc_converter.convert, url)
                        md_content = render.document.export_to_markdown()

                        markdown_output.append({
                            "url": url,
                            "title": entry.get("title"),
                            "markdown": md_content
                        })
                        url_memory.add(url)
                    except Exception as e:
                        print(f"Docling Error: {url} | {e}")
            except Exception as e:
                print(f"DDGS Search Error: {current_q} | {e}")
            finally:
                input_q.task_done()

    pool_size = min(3, len(target_queries))
    workers = [asyncio.create_task(worker_loop()) for _ in range(pool_size)]
    await input_q.join()

    for worker in workers:
        worker.cancel()

    with open(memory_path, "w") as f:
        json.dump(list(url_memory), f)

    print("-" * 40)
    print(f"Success: {len(markdown_output)} unique Markdown files generated.")
    print("=" * 40 + "\n")

    return {"markdown_pages": markdown_output}


# Node 5: Information Extraction
async def node_5_entity_extraction(state: HEXState) -> Dict[str, Any]:
    config = load_yaml("configs/config.yaml")
    pages_to_process = state.get("markdown_pages", [])


    if not pages_to_process:
        print("=" * 40)
        print("HEX STATE 5: ENTITY EXTRACTION")
        print("Warning: No markdown pages available for entity extraction.")
        print("=" * 40 + "\n")
        return {"extracted_entities": []}

    # 1. Load the external prompt file
    prompt_text = load_file("prompts/node5_prompt.txt")

    brain = ChatOpenAI(
        model=config['model_settings']['model_name'],
        temperature=0.0,
        api_key=openai_api_key
    )
    # The schema is enforced structurally here via Pydantic tool call binding
    structured_llm = brain.with_structured_output(EntityCollection)

    prompt = PromptTemplate.from_template(prompt_text)
    extraction_chain = prompt | structured_llm

    input_q = asyncio.Queue()
    for page in pages_to_process:
        await input_q.put(page)

    extracted_results = []

    print("=" * 40)
    print("HEX STATE 5: EXTRACTION")
    print(f"Status: Processing {len(pages_to_process)} complete webpages concurrently via schema.py...")
    print("-" * 40)

    async def extraction_worker():
        while not input_q.empty():
            page_data = await input_q.get()
            url = page_data.get("url", "Unknown")
            full_markdown = page_data.get("markdown", "")

            try:
                print(f"LLM-Extraction -> Parsing Entire Webpage: {url}")
                response: EntityCollection = await extraction_chain.ainvoke({
                    "user_query": state["user_query"],
                    "source_url": url,
                    "markdown_content": full_markdown
                })

                if response and response.entities:
                    for entity in response.entities:
                        entity_dict = entity.model_dump()
                        entity_dict["source_url"] = url
                        extracted_results.append(entity_dict)
            except Exception as e:
                print(f"Schema Extraction Error on complete file [{url}]: {e}")
            finally:
                input_q.task_done()

    concurrency_limit = min(3, len(pages_to_process))
    workers = [asyncio.create_task(extraction_worker()) for _ in range(concurrency_limit)]
    await input_q.join()

    for worker in workers:
        worker.cancel()

    unique_entities = []
    seen = set()
    for item in extracted_results:
        dedup_key = (item["service_name"].lower().strip(), item["location"].lower().strip())
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_entities.append(item)

    print("-" * 40)
    print(f"Success: Fully extracted {len(unique_entities)} distinct entities from raw data.")
    print("=" * 40 + "\n")

    return {"extracted_entities": unique_entities}


# Node 6: Verification Node
async def node_6_entity_verification(state: HEXState) -> Dict[str, Any]:
    config = load_yaml("configs/config.yaml")
    entities_to_verify = state.get("extracted_entities", [])

    if not entities_to_verify:
        print("=" * 40)
        print("HEX STATE 6: ENTITY VERIFICATION")
        print("Warning: No entities available in state for verification.")
        print("=" * 40 + "\n")
        return {"extracted_entities": []}

    # Read prompt files from outside the script
    criteria_desc = load_file("prompts/verification_criteria.txt").strip()
    task_query_desc = load_file("prompts/verification_task_query.txt").strip()
    system_prompt_template = load_file("prompts/verification_system_prompt.txt").strip()

    # Inject outside variables into the generalized template layout
    SYSTEM_PROMPT = system_prompt_template.format(
        criteria_desc=criteria_desc,
        task_query_desc=task_query_desc
    )

    # Initialize ChatOpenAI client with JSON object enforcement
    brain = ChatOpenAI(
        model=config['model_settings']['model_name'],
        temperature=0.0,
        api_key=openai_api_key
    ).bind(response_format={"type": "json_object"})

    input_q = asyncio.Queue()
    for entity in entities_to_verify:
        await input_q.put(entity)

    verified_results = []

    print("=" * 40)
    print("HEX STATE 6: GENERALIZED CONCURRENT VERIFICATION")
    print(f"Status: Validating {len(entities_to_verify)} records from state 5...")
    print("-" * 40)

    # Concurrent Worker Processing Loop
    async def verification_worker():
        while not input_q.empty():
            entity_data = await input_q.get()

            # Dynamically compile ALL fields passed from state 5 schema into user prompt text
            user_prompt_lines = ["Incoming Record Details:"]
            for field_key, field_val in entity_data.items():
                user_prompt_lines.append(f"{field_key}: {field_val}")
            user_prompt = "\n".join(user_prompt_lines)

            record_identifier = entity_data.get("service_name", entity_data.get("name", "Unknown Record"))

            try:
                print(f"LLM-Verify -> Processing Record: '{record_identifier}'")

                response = await brain.ainvoke([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ])

                verdict = json.loads(response.content)

                # Dynamic mapping of fields back into the record
                for verdict_key, verdict_val in verdict.items():
                    entity_data[verdict_key] = verdict_val

                verified_results.append(entity_data)

            except Exception as e:
                print(f"Verification Failure on entry [{record_identifier}]: {e}")
                entity_data["is_verified"] = 0
                entity_data["confidence_score"] = 0.0
                entity_data["reasoning"] = f"Pipeline Processing Error: {str(e)}"
                verified_results.append(entity_data)
            finally:
                input_q.task_done()
                await asyncio.sleep(0.1)

    concurrency_limit = min(3, len(entities_to_verify))
    workers = [asyncio.create_task(verification_worker()) for _ in range(concurrency_limit)]
    await input_q.join()

    for worker in workers:
        worker.cancel()

    # Save backup json lines file
    with open("verified_records.json", "w", encoding="utf-8") as f:
        json.dump(verified_results, f, indent=4, ensure_ascii=False)

    print("-" * 40)
    print(f"Success: Fully verified {len(verified_results)} records.")
    print("=" * 40 + "\n")

    # Direct downstream transfer override
    return {"extracted_entities": verified_results}



# Json to CSV Mapping

def csv_export(state: HEXState) -> Dict[str, Any]:
    # Look across state storage dynamically if explicit payload field was wiped
    verified_entities = state.get("extracted_entities", [])
    output_path = "output_records.csv"

    print("=" * 40)
    print("HEX STATE 7: STRUCTURAL CSV TRANSFORMATION")

    if not verified_entities:
        print("Error: No data available in execution state for CSV translation.")
        print("=" * 40 + "\n")
        return {"extracted_entities": []}

    print(f"Status: Structuring {len(verified_entities)} rows into columns...")
    print("-" * 40)

    # Dynamic Column Discovery Layout
    field_set = set()
    for entity in verified_entities:
        field_set.update(entity.keys())

    preferred_order = ["service_name", "location", "contact", "description", "source_url", "is_verified",
                       "confidence_score", "reasoning"]
    extra_fields = sorted(list(field_set - set(preferred_order)))
    final_column_headers = preferred_order + extra_fields

    try:
        with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=final_column_headers)
            writer.writeheader()
            for entity in verified_entities:
                row_data = {col: entity.get(col, "") for col in final_column_headers}
                writer.writerow(row_data)

        print(f"SUCCESS: Mapped JSON structural objects directly to spreadsheet format.")
        print(f"File Saved At Location -> {os.path.abspath(output_path)}")
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"Structural CSV Export Failure: {e}")
        print("=" * 40 + "\n")

    return {"extracted_entities": verified_entities}

# File System Cache Guards
def clear_cache_directory(directory_path: str = "./memory"):
    """
    Safely flushes and removes the local memory buffer cache directory
    before initiating an entirely new graph sequence.
    """
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Cache Guard: Successfully flushed cache directory '{directory_path}'.")
        except Exception as e:
            print(f"Cache Guard Warning: Could not clear directory '{directory_path}': {e}")



async def run_hex():
    clear_cache_directory("./memory")

    try:
        user_input_query = load_file("input/query.txt")

        inputs = {
            "user_query": user_input_query,
            "region_type": "",
            "city_names": [],
            "search_queries": [],
            "markdown_pages": [],
            "extracted_entities": []
        }

        # Keep a safe local memory register during streaming steps
        final_state_data = []

        print("Starting LangGraph Application Engine Workflow...")
        async for output in app.astream(inputs):
            for node_name, node_state in output.items():
                print(f"--- Finished Node: {node_name} ---")

                # Check for extracted items coming from either node 5 or node 6 updates
                if "extracted_entities" in node_state and node_state["extracted_entities"]:
                    final_state_data = node_state["extracted_entities"]

        # Double check generation safety if graph terminates cleanly
        if final_state_data:
            print("Execution stream finished processing records successfully.")
        else:
            print("Processing complete, but data array was empty across node steps.")

    except Exception as e:
        print(f"Execution failed: {e}")


# Routing logic
def route_decision(state: HEXState):
    if "macro" in state.get("region_type", ""):
        return "node_2"
    return "node_4"


#Langgraph Setup
workflow = StateGraph(HEXState)

# Add all 7 execution blocks inside the graph structure
workflow.add_node("analysis", task_analysis_node)
workflow.add_node("node_2", node_2_dbpedia)
workflow.add_node("node_3", node_3_query_reformulation)
workflow.add_node("node_4", node_4_ddgs_retrieval)
workflow.add_node("node_5", node_5_entity_extraction)
workflow.add_node("node_6", node_6_entity_verification)
workflow.add_node("node_7", csv_export)  # Added CSV Node 7

workflow.set_entry_point("analysis")

workflow.add_conditional_edges(
    "analysis",
    route_decision,
    {
        "node_2": "node_2",
        "node_4": "node_4"
    }
)

workflow.add_edge("node_2", "node_3")
workflow.add_edge("node_3", "node_4")
workflow.add_edge("node_4", "node_5")
workflow.add_edge("node_5", "node_6")
workflow.add_edge("node_6", "node_7")  # Connect verification directly to CSV export
workflow.add_edge("node_7", END)  # Final execution terminates at END node

app = workflow.compile()


if __name__ == "__main__":
    asyncio.run(run_hex())
