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


## Classical Sparse Retrieval


## Dense Retrieval


## Hybrid Retrieval


## Reranking with Cross Encoders


## Query Expansion and Self-Query


## Multi-hop Retrieval


## Evaluating Retriever in Isolation and In RAG


## Fine-Tuning Your Own Retriever


## Scaling and Deployment

