from flask import Flask, render_template, request, jsonify
import logging
import os
import json
import math
from typing import Dict, Union
from flask import send_file, send_from_directory
from gensim.models import KeyedVectors

app = Flask(__name__)
logging.basicConfig(
    handlers=[
        logging.FileHandler(__file__ + ".log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.DEBUG,
)

# -----------------------------------------------------------------------------
# Load GO/HP models from Models/ directory
# -----------------------------------------------------------------------------
MODELS_DIR = "./Updated_models"
ModelType = Union[KeyedVectors, Dict[str, list[float]]]
loaded_models: Dict[str, ModelType] = {}

logging.info(f"Loading embedding models from {MODELS_DIR}")
if os.path.isdir(MODELS_DIR):
    for fname in os.listdir(MODELS_DIR):
        path = os.path.join(MODELS_DIR, fname)
        name, ext = os.path.splitext(fname)
        try:
            if ext.lower() == ".kv":
                logging.info(f"  → Loading KV '{fname}'")
                loaded_models[name] = KeyedVectors.load(path, mmap="r")
            elif ext.lower() == ".json":
                logging.info(f"  → Loading JSON '{fname}'")
                with open(path, "r", encoding="utf-8") as f:
                    loaded_models[name] = json.load(f)
            else:
                logging.debug(f" — Skipping '{fname}'")
        except Exception as e:
            logging.error(f"Failed to load '{fname}': {e}")
else:
    logging.error(f"Models dir not found: {MODELS_DIR}")

logging.info(f"Total GO/HP models loaded: {len(loaded_models)}")

def cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(a*b for a,b in zip(v1,v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    return dot/(n1*n2) if n1 and n2 else 0.0


# -----------------------------------------------------------------------------
# UI routes (including your unified query.html with GO/HP tabs)
# -----------------------------------------------------------------------------
@app.route("/")
@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/query.html")
def query():
    return render_template("query.html", model_list=sorted(loaded_models.keys()))

@app.route("/licenses.html")
def licenses():
    return render_template("licenses.html")

@app.route("/download.html")
def download():
    return render_template("download.html")

@app.route("/contact.html")
def contact():
    return render_template("contact.html")


@app.route('/download_direct/<filename>')
def download_direct(filename):
    filepath = f"{filename}"
    return send_file(filepath, as_attachment=True)


@app.route('/download_folder/<ont>/<version>/<filename>')
def download_folder(ont, version, filename):
    base_dir = os.getcwd()
    filepath = os.path.join(base_dir, ont, version, filename)
    return send_file(filepath, as_attachment=True)


@app.route("/rest/models", methods=["GET"])
def rest_models():
    return jsonify(sorted(loaded_models.keys()))


# -----------------------------------------------------------------------------
# Similarty for GO
# -----------------------------------------------------------------------------

@app.route("/rest/calculate-similarity-go", methods=["POST"])
def rest_go():
    data = request.json or {}
    model, raw_id1, raw_id2 = data.get("model"), data.get("id1"), data.get("id2")
    model = "GO_" + model
    if model not in loaded_models:
        return jsonify(error="Model not found"), 404
    mod = loaded_models[model]
    
    if raw_id1.startswith("GO_"):
        id1 = "http://purl.obolibrary.org/obo/" + raw_id1
    else:
        id1 = loaded_models["GO_Labels"][raw_id1.lower().replace("_", " ")]

    if raw_id2.startswith("GO_"):
        id2 = "http://purl.obolibrary.org/obo/" + raw_id2
    else:
        id2 = loaded_models["GO_Labels"][raw_id2.lower().replace("_", " ")]
    
    if isinstance(mod, KeyedVectors):
        if id1 not in mod or id2 not in mod:
            return jsonify(error="GO IDs not found"), 400
        score = mod.similarity(id1, id2)
    else:
        v1, v2 = mod.get(id1), mod.get(id2)
        if v1 is None or v2 is None:
            return jsonify(error="GO IDs not found"), 400
        score = cosine(v1, v2)
    
    data = {"label1": loaded_models["GO_Labels"][id1] + " ["+ id1.split("/")[-1] + "]", 
            "label2": loaded_models["GO_Labels"][id2] + " ["+ id2.split("/")[-1] + "]", 
            "url1": id1, 
            "url2":  id2, 
            "similarity": float(score)}
    return jsonify(data)


# -----------------------------------------------------------------------------
# Similarty for HP
# -----------------------------------------------------------------------------

@app.route("/rest/calculate-similarity-hp", methods=["POST"])
def rest_hp():
    data = request.json or {}
    model, raw_id1, raw_id2 = data.get("model"), data.get("id1"), data.get("id2")
    model = "HP_" + model
    
    if raw_id1.startswith("HP_"):
        id1 = "http://purl.obolibrary.org/obo/" + raw_id1
    else:
        id1 = loaded_models["HP_Labels"][raw_id1.lower().replace("_", " ")]

    if raw_id2.startswith("HP_"):
        id2 = "http://purl.obolibrary.org/obo/" + raw_id2
    else:
        id2 = loaded_models["HP_Labels"][raw_id2.lower().replace("_", " ")]
    
    if model not in loaded_models:
        return jsonify(error="Model not found"), 404
    mod = loaded_models[model]
    if isinstance(mod, KeyedVectors):
        if id1 not in mod or id2 not in mod:
            return jsonify(error="HP IDs not found"), 400
        score = mod.similarity(id1, id2)
    else:
        v1, v2 = mod.get(id1), mod.get(id2)
        if v1 is None or v2 is None:
            return jsonify(error="HP IDs not found"), 400
        score = cosine(v1, v2)
    
    data = {"label1": loaded_models["HP_Labels"][id1] + " ["+ id1.split("/")[-1] + "]", 
            "label2": loaded_models["HP_Labels"][id2] + " ["+ id2.split("/")[-1] + "]", 
            "url1": id1, 
            "url2":  id2, 
            "similarity": float(score)}
    return jsonify(data)


# -----------------------------------------------------------------------------
# N-Closest neighbors for GO
# -----------------------------------------------------------------------------
@app.route("/rest/closest-go", methods=["POST"])
def closest_go():
    data = request.json or {}
    model = data.get("model")
    model = "GO_" + model
    raw_key = data.get("key")
    top_n = int(data.get("top_n", 10))

    if model not in loaded_models:
        return jsonify(error="Model not found"), 404

    mod = loaded_models[model]

    if raw_key.startswith("GO_"):
        key = "http://purl.obolibrary.org/obo/" + raw_key
    else:
        key = loaded_models["GO_Labels"][raw_key.lower().replace("_", " ")]

    vec = mod.get(key)
    if vec is None:
        return jsonify(error="ID or Label not found"), 400
    sims = []
    for k, v in mod.items():
        if k == key: continue
        sims.append((k, cosine(vec, v)))
    sims.sort(key=lambda x: x[1], reverse=True)
    neighbors = sims[:top_n]

    # Format response
    return jsonify([
        {"key": loaded_models["GO_Labels"][k] + " [" + k.split("/")[-1] + "]", "link":k, "similarity": float(score)}
        for k, score in neighbors
    ])


# -----------------------------------------------------------------------------
# N-Closest neighbors for HP
# -----------------------------------------------------------------------------
@app.route("/rest/closest-hp", methods=["POST"])
def closest_hp():
    data = request.json or {}
    model = data.get("model")
    model = "HP_" + model
    raw_key = data.get("key")
    top_n = int(data.get("top_n", 10))

    if model not in loaded_models:
        return jsonify(error="Model not found"), 404

    mod = loaded_models[model]

    if raw_key.startswith("HP_"):
        key = "http://purl.obolibrary.org/obo/" + raw_key
    else:
        key = loaded_models["HP_Labels"][raw_key.lower().replace("_", " ")]

    vec = mod.get(key)
    if vec is None:
        return jsonify(error="Key not found"), 400
    sims = []
    for k, v in mod.items():
        if k == key: continue
        sims.append((k, cosine(vec, v)))
    sims.sort(key=lambda x: x[1], reverse=True)
    neighbors = sims[:top_n]

    return jsonify([
        {"key": loaded_models["HP_Labels"][k] + " [" + k.split("/")[-1] + "]", "link":k, "similarity": float(score)}
        for k, score in neighbors
    ])


from gevent.pywsgi import WSGIServer
if __name__ == "__main__":
    
    #app.run(debug=True)
    
    http_server = WSGIServer(('127.0.0.1', 500), app)
    http_server.serve_forever()