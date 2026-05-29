# Baselines/PRF_baseline/prf_pipeline.py
import os
import re
import math
import yaml
import requests
from typing import TypedDict, List, Dict, Any
from bs4 import BeautifulSoup

# LangGraph Core Setup
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Shared Baseline Production Primitives
from ddgs import DDGS
from docling.document_converter import DocumentConverter

# Handle dynamic system path lookups safely
current_script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_script_dir, "config.yaml"), "r", encoding="utf-8") as y_f:
    CONFIG = yaml.safe_load(y_f)


class PrfState(TypedDict):
    task_query: str
    initial_query: str
    initial_snippets: List[Dict[str, str]]
    pseudo_relevant_text: str
    feedback_terms: str
    final_expanded_query: str
    discovered_urls: List[str]
    raw_markdown_corpus: str
    extracted_table: List[Dict[str, Any]]


def extract_top_tfidf_terms(documents: List[str], top_k: int) -> str:
    """Computes TF-IDF weights over pseudo-relevant snippets to pull salient terms."""

    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z]{4,15}\b', text.lower())

    stop_words = {'with', 'from', 'that', 'this', 'your', 'located', 'services', 'find', 'list', 'directory', 'ontario',
                  'toronto', 'health', 'care', 'hospitals'}
    doc_tokens = [list(set(tokenize(d)) - stop_words) for d in documents if d]

    if not doc_tokens or not any(doc_tokens):
        return "directory listings lookup"

    df = {}
    for doc in doc_tokens:
        for word in doc:
            df[word] = df.get(word, 0) + 1

    vocab_scores = {}
    total_docs = len(doc_tokens)
    for doc in doc_tokens:
        for word in doc:
            tf = doc.count(word)
            idf = math.log((1 + total_docs) / (1 + df[word])) + 1
            vocab_scores[word] = vocab_scores.get(word, 0) + (tf * idf)

    sorted_terms = sorted(vocab_scores.items(), key=lambda x: x, reverse=True)
    return " ".join([term for term, score in sorted_terms[:top_k]])


def initial_rewrite_node(state: PrfState) -> dict:
    print(f"\n[PRF INITIATION] Base Task Query: '{state['task_query']}'")
    p_path = os.path.join(current_script_dir, CONFIG["paths"]["initial_rewrite_prompt"])
    with open(p_path, "r", encoding="utf-8") as f:
        prompt_skeleton = f.read()

    llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])
    response = llm.invoke([HumanMessage(content=prompt_skeleton.format(task_query=state["task_query"]))])
    return {"initial_query": response.content.strip().replace('"', '')}


def round_one_retrieval(state: PrfState) -> dict:
    query = state["initial_query"]
    max_res = int(CONFIG["tools"]["search_max_results"])
    print(f"[PRF ROUND 1 SEARCH] Querying: '{query}'")
    snippets = []
    try:
        results = DDGS().text(query, max_results=max_res)
        if results:
            for r in results:
                snippets.append({"url": r.get("href", ""), "text": f"{r.get('title', '')} {r.get('body', '')}"})
    except Exception as e:
        print(f"[PRF ERROR] Round 1 search failure: {str(e)}")
    return {"initial_snippets": snippets}


def relevance_judge_node(state: PrfState) -> dict:
    print("[PRF RELEVANCE JUDGE] Evaluating document snippet matrices...")
    j_path = os.path.join(current_script_dir, CONFIG["paths"]["judge_prompt"])
    with open(j_path, "r", encoding="utf-8") as f:
        judge_skeleton = f.read()

    llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])
    relevant_blocks = []
    threshold = float(CONFIG["tools"]["relevance_threshold"])

    for item in state["initial_snippets"]:
        try:
            eval_prompt = judge_skeleton.format(task_query=state["task_query"], snippet=item["text"])
            score_res = llm.invoke([HumanMessage(content=eval_prompt)]).content.strip()
            score = float(re.findall(r"[0-9.]+", score_res))

            if score >= threshold:
                relevant_blocks.append(item["text"])
        except Exception:
            continue

    print(f"[PRF JUDGE COMPLETED] Identified {len(relevant_blocks)} pseudo-relevant contexts.")
    return {"pseudo_relevant_text": " ".join(relevant_blocks)}


def tfidf_feedback_node(state: PrfState) -> dict:
    print("[PRF TF-IDF] Extracting salient context expansion tokens...")
    doc_corpus = [state["pseudo_relevant_text"]] if state["pseudo_relevant_text"] else []
    top_terms = extract_top_tfidf_terms(doc_corpus, int(CONFIG["tools"]["top_k_feedback_terms"]))
    print(f"[PRF TF-IDF SUCCESS] Feedback expansion keys: '{top_terms}'")
    return {"feedback_terms": top_terms}


def feedback_reformulation_node(state: PrfState) -> dict:
    f_path = os.path.join(current_script_dir, CONFIG["paths"]["feedback_rewrite_prompt"])
    with open(f_path, "r", encoding="utf-8") as f:
        skeleton = f.read()

    llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])
    prompt = skeleton.format(task_query=state["task_query"], feedback_terms=state["feedback_terms"])
    response = llm.invoke([HumanMessage(content=prompt)])
    expanded_query = response.content.strip().replace('"', '')
    print(f"[PRF REWRITER 2] Formulated secondary lookup query: '{expanded_query}'")
    return {"final_expanded_query": expanded_query}


def round_two_retrieval(state: PrfState) -> dict:
    query = state["final_expanded_query"]
    max_res = int(CONFIG["tools"]["search_max_results"])
    print(f"[PRF ROUND 2 SEARCH] Querying expansion string: '{query}'")
    urls = []
    try:
        results = DDGS().text(query, max_results=max_res)
        if results:
            urls = [r.get("href") for r in results if r.get("href")]
            print(f"[PRF ROUND 2 RETRIEVAL] Discovered {len(urls)} fresh targets.")
    except Exception as e:
        print(f"[PRF ERROR] Round 2 execution error: {str(e)}")
    return {"discovered_urls": urls}


def iterative_fetch_convert_and_extract_node(state: PrfState) -> dict:
    urls = state["discovered_urls"]
    accumulated_table_rows = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."}

    r_path = os.path.join(current_script_dir, CONFIG["paths"]["reader_prompt"])
    with open(r_path, "r", encoding="utf-8") as f:
        reader_skeleton = f.read()

    llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])

    # Import validation container wrapper schema dynamically
    from schema import EntityTable
    structured_llm = llm.with_structured_output(EntityTable)

    for idx, url in enumerate(urls):
        print(f"\n--- [PAGE RUN {idx + 1}/{len(urls)}] Processing Source: {url} ---")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            temp_html_path = os.path.join(current_script_dir, f"temp_prf_{os.getpid()}.html")
            with open(temp_html_path, "w", encoding="utf-8") as f:
                f.write(str(soup))

            print("[DOCLING CONVERTER] Generating layout-preserved structural markdown...")
            converter = DocumentConverter()
            docling_result = converter.convert(temp_html_path)
            markdown_text = docling_result.document.export_to_markdown()
            os.remove(temp_html_path)

            print("[PYDANTIC EXTRACTOR] Isolating structured rows from this webpage target...")
            system_prompt = reader_skeleton.format(task_query=state["task_query"], raw_markdown_content=markdown_text)

            extracted_output = structured_llm.invoke(system_prompt)
            page_rows = [row.model_dump() for row in extracted_output.rows]
            print(f"[PAGE EXTRACTED SUCCESS] Captured {len(page_rows)} structured records from this domain.")

            accumulated_table_rows.extend(page_rows)
        except Exception as e:
            print(f"[PRF WARNING] Pipeline skipped URL {url}: {str(e)}")
            continue

    print(f"\n[PRF TOTAL COMPLETE] Total aggregated tabular elements compiled: {len(accumulated_table_rows)}")
    return {"extracted_table": accumulated_table_rows}


def compile_prf_baseline():
    workflow = StateGraph(PrfState)
    workflow.add_node("rewrite_1", initial_rewrite_node)
    workflow.add_node("retrieve_1", round_one_retrieval)
    workflow.add_node("judge", relevance_judge_node)
    workflow.add_node("tfidf", tfidf_feedback_node)
    workflow.add_node("rewrite_2", feedback_reformulation_node)
    workflow.add_node("retrieve_2", round_two_retrieval)
    workflow.add_node("convert_and_read", iterative_fetch_convert_and_extract_node)

    workflow.set_entry_point("rewrite_1")
    workflow.add_edge("rewrite_1", "retrieve_1")
    workflow.add_edge("retrieve_1", "judge")
    workflow.add_edge("judge", "tfidf")
    workflow.add_edge("tfidf", "rewrite_2")
    workflow.add_edge("rewrite_2", "retrieve_2")
    workflow.add_edge("retrieve_2", "convert_and_read")
    workflow.add_edge("convert_and_read", END)

    # Locked: Explicitly returns compiled runnable graph
    return workflow.compile()
