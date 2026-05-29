# Baselines/R3/r3_pipeline.py
import os
import yaml
import requests
from typing import TypedDict, List, Dict, Any
from bs4 import BeautifulSoup

# LangGraph Core Components
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Shared Baseline Production Primitives
from ddgs import DDGS
from docling.document_converter import DocumentConverter
from prompts.schema import EntityTable

# Load system configuration keys safely
current_script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config_path = os.path.join(current_script_dir, "config.yaml")

with open(yaml_config_path, "r", encoding="utf-8") as y_f:
    CONFIG = yaml.safe_load(y_f)


# A. PIPELINE STATE DEFINITION
class R3State(TypedDict):
    task_query: str
    rewritten_query: str
    discovered_urls: List[str]
    raw_markdown_corpus: str
    extracted_table: List[Dict[str, Any]]


# B. STATIC OPERATIONAL BLOCK NODES
def rewrite_query_node(state: R3State) -> dict:
    """Implements R3's EMNLP few-shot query reformulation technique."""
    print(f"\n[R3 REWRITE] Inputting original task query: '{state['task_query']}'")

    prompt_path = os.path.join(current_script_dir, CONFIG["paths"]["rewrite_prompt_file"])
    with open(prompt_path, "r", encoding="utf-8") as f:
        rewrite_skeleton = f.read()

    formatted_prompt = rewrite_skeleton.format(task_query=state["task_query"])

    llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    rewritten_string = response.content.strip().replace('"', '')
    print(f"[R3 REWRITE SUCCESS] Reformulated search query payload: '{rewritten_string}'")
    return {"rewritten_query": rewritten_string}


def execute_discovery_search(state: R3State) -> dict:
    """Queries Dux Distributed Global Search (DDGS) using the rewritten string."""
    query = state["rewritten_query"]
    max_results = int(CONFIG["tools"]["search_max_results"])
    print(f"[R3 RETRIEVAL] Issuing global search for: '{query}'")

    urls = []
    try:
        results = DDGS().text(query, max_results=max_results)
        if results:
            urls = [r.get("href") for r in results if r.get("href")]
            print(f"[R3 RETRIEVAL] Found {len(urls)} target candidate URLs.")
    except Exception as e:
        print(f"[R3 ERROR] Search discovery retrieval failure: {str(e)}")

    return {"discovered_urls": urls}


def fetch_and_convert_to_markdown(state: R3State) -> dict:
    """Downloads target document HTML targets and transforms them into clean Markdown via Docling."""
    urls = state["discovered_urls"]
    compiled_markdown_blocks = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for idx, url in enumerate(urls):
        print(f"[R3 READ & FLATTEN] ({idx + 1}/{len(urls)}) Converting layout files: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            temp_html_path = os.path.join(current_script_dir, f"temp_r3_{os.getpid()}.html")
            with open(temp_html_path, "w", encoding="utf-8") as f:
                f.write(str(soup))

            converter = DocumentConverter()
            docling_result = converter.convert(temp_html_path)
            markdown_text = docling_result.document.export_to_markdown()
            os.remove(temp_html_path)

            markdown_block = f"--- START DOCUMENT: {url} ---\n{markdown_text}\n--- END DOCUMENT ---\n"
            compiled_markdown_blocks.append(markdown_block)
        except Exception as e:
            print(f"[R3 WARNING] Failed to digest URL {url}: {str(e)}")

    aggregated_context = "\n".join(compiled_markdown_blocks)
    return {"raw_markdown_corpus": aggregated_context}


def single_turn_reader_extraction(state: R3State) -> dict:
    """Reads aggregated markdown data streams to isolate rows using schema boundaries."""
    print("\n[R3 READER] Executing single-turn extraction via schema context map...")
    aggregated_markdown = state["raw_markdown_corpus"]

    if not aggregated_markdown.strip():
        print("[R3 WARNING] Context stream is completely empty. Yielding empty list.")
        return {"extracted_table": []}

    prompt_path = os.path.join(current_script_dir, CONFIG["paths"]["reader_prompt_file"])
    with open(prompt_path, "r", encoding="utf-8") as f:
        reader_skeleton = f.read()

    try:
        system_prompt = reader_skeleton.format(
            task_query=state["task_query"],
            raw_markdown_content=aggregated_markdown
        )

        llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])
        structured_llm = llm.with_structured_output(EntityTable)

        extracted_output = structured_llm.invoke(system_prompt)
        serialized_rows = [row.model_dump() for row in extracted_output.rows]

        print(f"[R3 READER SUCCESS] Successfully parsed {len(serialized_rows)} elements into table.")
        return {"extracted_table": serialized_rows}
    except Exception as e:
        print(f"[R3 READER CRASH] Extraction engine structural failure: {str(e)}")
        return {"extracted_table": []}


# C. PIPELINE GRAPH ASSEMBLER
def compile_r3_baseline():
    workflow = StateGraph(R3State)

    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("retrieve", execute_discovery_search)
    workflow.add_node("convert", fetch_and_convert_to_markdown)
    workflow.add_node("read", single_turn_reader_extraction)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "convert")
    workflow.add_edge("convert", "read")
    workflow.add_edge("read", END)

    return workflow.compile()
