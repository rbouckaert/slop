from flask import Flask, render_template, request, jsonify
from Bio import Phylo
import io
import math
import json

app = Flask(__name__)

def jukes_cantor_matrix(t, mu=1.0, k=4):
    # Use a small default if branch length is missing to avoid zero-length errors
    t = float(t) if t is not None else 0.1
    p_same = (1/k) + ((k-1)/k) * math.exp(-(k/(k-1)) * mu * t)
    p_diff = (1/k) - (1/k) * math.exp(-(k/(k-1)) * mu * t)
    return p_same, p_diff

def calculate_likelihoods(node, tip_states, states_map):
    k = len(states_map)
    partials = [0.0] * k
    
    if node.is_terminal():
        state = tip_states.get(node.name)
        if state in states_map:
            partials[states_map[state]] = 1.0
        else:
            for i in range(0,4):
                partials[i] = 1.0
        node.partials = partials
        return partials

    children_partials = []
    for child in node.clades:
        child_p = calculate_likelihoods(child, tip_states, states_map)
        p_same, p_diff = jukes_cantor_matrix(child.branch_length)
        
        integrated_p = []
        for parent_s in range(k):
            sum_p = 0
            for child_s in range(k):
                prob_trans = p_same if parent_s == child_s else p_diff
                sum_p += prob_trans * child_p[child_s]
            integrated_p.append(sum_p)
        children_partials.append(integrated_p)

    node_partials = [1.0] * k
    for i in range(k):
        for cp in children_partials:
            node_partials[i] *= cp[i]
    
    s = sum(node_partials)
    node.partials = [p/s for p in node_partials] if s > 0 else node_partials
    return node_partials

def tree_to_dict(node, tip_data, dist=0):
    """Recursively build dict, tracking cumulative distance from root."""
    branch_len = node.branch_length if node.branch_length is not None else 0
    cumulative_dist = dist + branch_len
    
    return {
        "name": node.name if node.name else "",
        "dist": cumulative_dist,
        "branch_length": branch_len,
        "partials": tip_data.get(node.name) if node.is_terminal() else getattr(node, 'partials', []),
        "children": [tree_to_dict(c, tip_data, cumulative_dist) for c in node.clades]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    nwk = request.form.get('newick')
    tip_data = json.loads(request.form.get('states'))
    states_list = ['A', 'C', 'G', 'T']
    states_map = {s: i for i, s in enumerate(states_list)}
    
    tree = next(Phylo.parse(io.StringIO(nwk), "newick"))
    calculate_likelihoods(tree.root, tip_data, states_map)
    
    # Pass 0 as the starting distance for the root
    tree2 = tree_to_dict(tree.root, tip_data, 0)
    print(tree2)
    return jsonify(tree_to_dict(tree.root, tip_data, 0))

if __name__ == '__main__':
    app.run(debug=True)