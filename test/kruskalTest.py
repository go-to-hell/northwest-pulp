import networkx as nx

def create_graph(data):
    G = nx.Graph()
    for node in data['nodes'].values():
        G.add_node(node['id'])
    for edge_id, edge in data['edges'].items():
        G.add_edge(edge['source'], edge['target'], weight=int(edge['label']), id=edge_id)
    return G

def create_data(G):
    edges = {edge[2]['id']: {"source": edge[0], "target": edge[1], "label": str(edge[2]['weight'])} for edge in G.edges(data=True)}
    return {"edges": edges}

def find_spanning_tree(data, maximize=False):
    G = create_graph(data)
    T = nx.maximum_spanning_tree(G, weight='weight') if maximize else nx.minimum_spanning_tree(G, weight='weight')
    return create_data(T)

def create_paths(data):
    paths = {f"path{i+1}": {"edges": [edge]} for i, edge in enumerate(data['edges'].keys())}
    return paths

data = {
  "nodes": {
    "node1": {
      "id": "node1",
      "name": "Nodo 1",
      "x": -374,
      "y": 5
    },
    "node2": {
      "id": "node2",
      "name": "Nodo 2",
      "x": -167,
      "y": -150
    },
    "node3": {
      "id": "node3",
      "name": "Nodo 3",
      "x": -163,
      "y": 66
    },
    "node4": {
      "id": "node4",
      "name": "Nodo 4",
      "x": 55,
      "y": -167
    },
    "node5": {
      "id": "node5",
      "name": "Nodo 5",
      "x": 58,
      "y": 53
    },
    "node6": {
      "id": "node6",
      "name": "Nodo 6",
      "x": 220,
      "y": -169
    },
    "node7": {
      "id": "node7",
      "name": "Nodo 7",
      "x": 242,
      "y": 50
    },
    "node8": {
      "id": "node8",
      "name": "Nodo 8",
      "x": 409,
      "y": -23
    }
  },
  "edges": {
    "edge1": {
      "source": "node1",
      "target": "node2",
      "label": 6
    },
    "edge2": {
      "source": "node1",
      "target": "node3",
      "label": 2
    },
    "edge3": {
      "source": "node2",
      "target": "node3",
      "label": 3
    },
    "edge4": {
      "source": "node2",
      "target": "node4",
      "label": 4
    },
    "edge5": {
      "source": "node2",
      "target": "node5",
      "label": 1
    },
    "edge6": {
      "source": "node4",
      "target": "node6",
      "label": 3
    },
    "edge7": {
      "source": "node5",
      "target": "node6",
      "label": 6
    },
    "edge8": {
      "source": "node4",
      "target": "node5",
      "label": 7
    },
    "edge9": {
      "source": "node3",
      "target": "node5",
      "label": 4
    },
    "edge10": {
      "source": "node5",
      "target": "node7",
      "label": 9
    },
    "edge11": {
      "source": "node6",
      "target": "node7",
      "label": 7
    },
    "edge12": {
      "source": "node6",
      "target": "node8",
      "label": 2
    },
    "edge13": {
      "source": "node7",
      "target": "node8",
      "label": 4
    }
  },
}

data_mst = find_spanning_tree(data, maximize=False)
paths = create_paths(data_mst)
print(data_mst)
print(paths)