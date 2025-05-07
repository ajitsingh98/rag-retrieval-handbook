# Retrieval Strategies in RAG

## Table of Contents:

- [Introduction](#introduction)
- [Dataset Info](#dataset-info)
- [Why Retrieval at All?](#why-retrieval-at-all)
- [Classical Sparse Retrieval](#classical-sparse-retrieval)
- [Dense Retrieval](#dense-retrieval)
- [Hybrid Retrieval](hybrid-retrieval)
- [Reranking with Cross Encoders](#reranking-with-cross-encoders)
- [Query Expansion and Self-Query](#query-expansion-and-self-query)
- [Multi-hop Retrieval](#multi-hop-retrieval)
- [Evaluating Retriever in Isolation and In RAG](#evaluating-retriever-in-isolation-and-in-rag)
- [Fine-Tuning Your Own Retriever](#fine-tuning-your-own-retriever)
- [Scaling and Deployment](#scaling-and-deployment)

## Introduction


## Dataset Info

### HotpotQA

A Dataset for Diverse and Explainable Multi-hop Question Answering

**Source:** [HotpotQA](https://huggingface.co/papers/1809.09600) Paper on Hugging Face

HotpotQA is a large-scale question answering dataset designed to support multi-hop reasoning and explainability in QA systems. It consists of $113,000$ question-answer pairs grounded in Wikipedia content. This dataset addresses key limitations in existing QA benchmarks by incorporating the following characteristics:

- **Multi-hop Reasoning:** Each question requires retrieving and synthesizing information from multiple supporting documents to arrive at the correct answer.
- **Diverse Question Types:** The questions span a wide range of topics and are not restricted to predefined knowledge bases or ontologies.
- **Sentence-level Supervision:** Annotated supporting facts are provided at the sentence level, enabling models to learn interpretable reasoning paths and generate explainable predictions.
- **Factoid Comparison Questions:** Includes a novel category of comparison questions that assess a model’s ability to identify and contrast relevant facts.

HotpotQA significantly challenges state-of-the-art QA systems, making it an ideal benchmark for evaluating retrieval and reasoning capabilities in RAG pipelines.


## Why Retrieval at All?

### From Closed book to open book

#### Closed-book LLM

- Closed-book means the model answers using only the parametric knowledge stored into its weights.
- No external source is provided at the inference time

**Strengths**

- Fast : Single forward pass
- Works offline

**Weakness**

- Knowledge cutoff (e.g GPT-2 knows nothing beyond 2019)
- Static: Updating facts => retrain or fine-tune
- Hallucinations: Will confidently fabricate plausible-looking but wrong facts
- Context window is wasted repeating facts the model already "knows" but may recall incorrectly 

#### Open-book LLM

- Closed-book LLM + Retriever 
- Inject fresh, task specific snippets (paragraphs, tables, code,..) into the prompt on-the-fly
- Mechanism = Retrieve k pieces of evidence -> Feed them, plus the user query, into the generator

**Benefits**

- Freshness: Index can be updated Independently 
- Compact models can punch above their weight because "knowledge" is off-loaded
- Controllability & Traceability: Can show citations and help in building trust

### Anatomy of minimal RAG Pipeline

- Step 0(Optional): *Query re-writer* reformulates noisy user text into canonical search query
- Step 1: *Retriever* (fast) given the query, find the top-k candidate chunk from a large corpus 
    - *Types*: Sparse(BM25), Dense (DPR/BGE), hybrid, multi-vector (ColBERT)
- Step 2: *Re-ranker(Slow and optional)* Cross encoder scores (query <-> chunk) to reorder top-n with finer semantics
- Step 3: *Prompt Assembler* Select <= context_window tokens, interleave citations, maybe add system instructions
- Step 4: *Generator* Any seq-to-seq or chat-LLM(eg. Llama-3-Instruct, Mixtral-8x7B, GPT-4)
- Step 5: *Post-processor(Optional)* Extract structured answer, highlight sources and detect Hallucinations 

**Key Interface Contract:**

- Retriever guarantees: The answer is likely in these k snippets 
- Generator guarantees: Given that atleast one snippet contains the answer, I can synthesize a coherent reply

### Why Not Just Make a Bigger Model?

1. **Cost**
    - GPT-4-class models -> Tens of millions $ to train 
    - ~15 GB VRAM to run at 8-bit
2. **Latency**
    - Embedding lookup in FAISS or a vector DB = a few ms
    - Cheaper than extra transformer layers
3. **Data Privacy & Custom Corpora**
    - Want the model to answer from your PDF collection? Retriever is the only sane path
4. **Explainability**
    - Citations enable audits and regulatory compliance (EU AI Act, HIPPA, ..)

### Evaluation Metrics in RAG System

- Retrieval Level
    - Recall@k (answer ∈ top-k)
    - MRR(Mean Reciprocal Rank (position of first hit))
    - Latency p50/p95 search time

- End-to-end
    - Exact Match (EM)
    - F1 token overlap
    - Helpful-Harmless-Honest(HHH)
    - BLEU/ROUGE for summarization tasks
    - Hallucination Rate (Answer citing no retrieved snippet)

## Classical Sparse Retrieval


## Dense Retrieval


## Hybrid Retrieval


## Reranking with Cross Encoders


## Query Expansion and Self-Query


## Multi-hop Retrieval


## Evaluating Retriever in Isolation and In RAG


## Fine-Tuning Your Own Retriever


## Scaling and Deployment

