#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import csv
import json
import pandas as pd
import torch
from rdflib import Graph, Namespace, RDF, RDFS
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def load_ontology(source):
    """Loads an ontology from a URL or local file."""
    g = Graph()
    g.parse(source, format="xml")
    return g


def extract_relationships(graph, relationship_csv):
    """Extracts relationships (edges) and saves them."""
    relationships = []
    for s, p, o in graph.triples((None, RDFS.subClassOf, None)):
        relationships.append([str(s), str(p), str(o)])
    df = pd.DataFrame(relationships, columns=["source", "relation", "target"])
    df.to_csv(relationship_csv, sep="\t", index=False, header=False)
    print(f"Saved KG triples to {relationship_csv}")


def run_embeddings(path_ontology, relationship_csv, models_to_train, output_dir):
    if not os.path.exists(relationship_csv):
        graph = load_ontology(path_ontology)
        extract_relationships(graph, relationship_csv)

    df_triples = pd.read_csv(relationship_csv, sep="\t", names=["head", "relation", "tail"])
    triples = df_triples[["head", "relation", "tail"]].values
    dataset = TriplesFactory.from_labeled_triples(triples)

    # Create directory to save embeddings
    results = {}
    for model_name in models_to_train:
        print(f"\nTraining {model_name}")
        result = pipeline(
            model=model_name,
            training = dataset,
            testing = dataset,
            training_loop="lcwa",
            epochs=100,
            model_kwargs={"embedding_dim": 200},
            use_tqdm=True
        )
        
        results[model_name] = result
        model = result.model

        # Get PyKEEN embedding objects
        entity_embeddings = model.entity_representations[0]
        relation_embeddings = model.relation_representations[0]
        
        # Retrieve mapping from entity/relation to internal IDs
        # Option 1: Use your original TriplesFactory from the dataset
        entity_to_id = dataset.entity_to_id
        relation_to_id = dataset.relation_to_id

        # Convert entity embeddings to a dictionary
        entity_dict = {}
        for entity, idx in entity_to_id.items():
            embedding_tensor = entity_embeddings(torch.tensor([idx]))
            entity_dict[entity] = embedding_tensor.detach().cpu().numpy().tolist()[0]
    
        # Convert relation embeddings to a dictionary
        relation_dict = {}
        for relation, idx in relation_to_id.items():
            embedding_tensor = relation_embeddings(torch.tensor([idx]))
            relation_dict[relation] = embedding_tensor.detach().cpu().numpy().tolist()[0]

        # ALSO Save the dictionaries to JSON
        with open(os.path.join(output_dir, f"{model_name}_entity.json"), "w") as f:
            json.dump(entity_dict, f, indent=4)
        with open(os.path.join(output_dir, f"{model_name}_relation.json"), "w") as f:
            json.dump(relation_dict, f, indent=4)

        print(f"Saved {model_name} embeddings as both pickle and JSON dictionaries!")



if __name__ == "__main__":

    embedding_models = ["BoxE", "distMult", "HoLE", "TransE", "TransR"]

    #GO
    versions = ["2025-06-01", "2024-11-03", "2024-06-10", "2024-01-17", "2023-06-11", "2023-01-01"]
    for version in versions:
        path_ontology = "./GO/" + version + "/go.owl"
        output_dir = "./GO/" + version
        relationship_csv = "./GO/" + version + "/relationships.tsv"
        run_embeddings(path_ontology, relationship_csv, embedding_models, output_dir)

    #HP
    versions = ["2025-05-06", "2025-01-16", "2024-07-01", "2024-01-16", "2023-06-17", "2023-01-27"]
    for version in versions:
        path_ontology = "./HP/" + version + "/hp.owl"
        output_dir = "./HP/" + version
        relationship_csv = "./HP/" + version + "/relationships.tsv"
        run_embeddings(path_ontology, relationship_csv, embedding_models, output_dir)
