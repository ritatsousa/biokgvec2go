import os
import json
import traceback
from rdflib import Graph
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

VECTOR_SIZE = 200
EPOCHS = 100

def log(msg):
    print(msg)
    with open("output.log", "a") as f:
        f.write(msg + "\n")

def check_path(p):
    if os.path.exists(p):
        log(f"Path exists: {p}")
    else:
        log(f"Path does NOT exist: {p}")
        raise FileNotFoundError(p)

def extract_entities_rdf(file_path, prefix):
    g = Graph()
    g.parse(file_path)
    log(f"Parsed {file_path} for entity extraction, triples: {len(g)}")
    entities = set()
    for s, p, o in g:
        for node in (s, o):
            uri = str(node)
            if uri.startswith("http") and prefix in uri:
                entities.add(uri)
    entities = list(entities)
    log(f"Extracted {len(entities)} unique entities from {prefix}")
    print(f"Sample extracted {prefix} entities:", entities[:5])
    return entities

def train_and_embed(kg_file, entities, label):
    log(f"Creating KG object for pyRDF2Vec ({label})...")
    kg = KG(kg_file, is_remote=False)
    log(f"KG created for {label}: {kg}")

    log(f"Training embeddings for {len(entities)} {label} entities. VECTOR_SIZE={VECTOR_SIZE}, EPOCHS={EPOCHS}")
    transformer = RDF2VecTransformer(
        Word2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS),
        walkers=[RandomWalker(4, 100)]
    )
    embeddings, literals = transformer.fit_transform(kg, entities)
    log(f"Training completed for {label}.")
    print(f"len({label} embeddings):", len(embeddings))
    print(f"Sample {label} embedding[0]:", embeddings[0] if embeddings else "EMPTY")
    return embeddings


def run_embeddings(ontology, ontology_file, output_file):
    try:
        log(f"==== Current working directory: {os.getcwd()} ====")
        log("==== Checking input file existence... ====")
        check_path(ontology_file)

        # Extract entities
        log("Extracting entities from ontology...")
        if ontology == "GO":
            entities = extract_entities_rdf(ontology_file, "GO_")
        elif ontology == "HP":
            entities = extract_entities_rdf(ontology_file, "HP_")

        # Train embeddings 
        embeddings = train_and_embed(ontology_file, entities, ontology)

        # Save embeddings as JSON
        output = {}
        for ent, emb in zip(entities, embeddings):
            output[str(ent)] = [float(x) for x in list(emb)]
        log(f"Output dictionary prepared")

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        log(f"Embeddings saved to {output_file}")

    except Exception as e:
        log("ERROR: Exception occurred!")
        log(traceback.format_exc())


if __name__ == "__main__":


    #GO
    versions = ["2025-06-01", "2024-11-03", "2024-06-10", "2024-01-17", "2023-06-11", "2023-01-01"]
    for version in versions:
        ontology_file = "./GO/" + version + "/go.owl"
        output_file = "./GO/" + version + "/RDF2Vec.json"
        run_embeddings("GO", ontology_file, output_file)

    #HP
    versions = ["2025-05-06", "2025-01-16", "2024-07-01", "2024-01-16", "2023-06-17", "2023-01-27"]
    for version in versions:
        ontology_file = "./HP/" + version + "/hp.owl"
        output_file = "./HP/" + version + "/RDF2Vec.json"
        run_embeddings("HP", ontology_file, output_file)
    
    