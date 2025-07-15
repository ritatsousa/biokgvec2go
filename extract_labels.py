from rdflib import Graph, RDF, RDFS
import json

def extract_iri_label_dict(owl_file, prefix, path_output):
    g = Graph()
    g.parse(owl_file, format='xml')

    dic_labels = {}
    for s, p, o in g.triples((None, RDFS.label, None)):
        if str(s).startswith("http") and prefix in str(s):
            standard_o = str(o).lower().replace("_", " ")
            dic_labels[standard_o] = str(s)
            dic_labels[str(s)] = standard_o

    with open(path_output, "w") as f:
        json.dump(dic_labels, f, indent=2)


ontology_file = "./GO/2025-06-01/go.owl"
output_file = "./GO/2025-06-01/GO_Labels.json"
extract_iri_label_dict(ontology_file, "GO_", output_file)

ontology_file = "./HP/2025-05-06/hp.owl"
output_file = "./HP/2025-05-06/HP_Labels.json"
extract_iri_label_dict(ontology_file, "HP_",  output_file)