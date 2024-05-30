#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/5/30 21:19
"""
import logging
import pickle

from tree2retriever.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from tree2retriever.embedding import SBertEmbeddingModel
from tree2retriever.summary import GLMSummarizationModel
from tree2retriever.tree_retriever import TreeRetriever, TreeRetrieverConfig

if __name__ == '__main__':
    tree_builder_type = 'cluster'
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

    tree_builder_class, tree_builder_config_class = supported_tree_builders[
        tree_builder_type
    ]
    embedding_model = SBertEmbeddingModel(model_name=r"I:\pretrained_models\bert\english\all-mpnet-base-v2")
    summary_model = GLMSummarizationModel(model_name_or_path=r"I:\pretrained_models\llm\chatglm3-6b")
    tree_builder_config = tree_builder_config_class(
        tokenizer=None,
        max_tokens=100,
        num_layers=5,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=100,
        summarization_model=summary_model,
        embedding_models={'sbert': embedding_model},
        cluster_embedding_model="sbert",
    )

    tree_retriever_config = TreeRetrieverConfig(
        tokenizer=None,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        context_embedding_model="sbert",
        embedding_model=embedding_model,
        num_layers=None,
        start_layer=None,
    )

    tree_builder = tree_builder_class(tree_builder_config)

    with open(r'H:\Projects\GoMate\data\docs\sample.txt', 'r') as file:
        text = file.read()
    tree = tree_builder.build_from_text(text=text)
    retriever = TreeRetriever(tree_retriever_config, tree)
    question = '"How did Cinderella reach her happy ending?'

    search_docs = retriever.retrieve(question)
    print(search_docs)

    path = "tree.pkl"
    with open(path, "wb") as file:
        pickle.dump(tree, file)
    logging.info(f"Tree successfully saved to {path}")
