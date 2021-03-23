from xir import Graph
import json
import sys

if len(sys.argv) < 3:
    print('usage: python disable_dru.py src.xmodel dst.xmodel')

def get_graph(p):

    graph = Graph.deserialize(p)
    subgraph = graph.get_root_subgraph()
    subcpu = 0
    subdpu = 0
    for child in subgraph.get_children():
        if child.get_attr('device') == 'DPU':
            subdpu = child
        elif child.get_attr('device') == 'CPU':
            subcpu = child
        else:
            subuser = child
    return graph, subgraph, subdpu, subcpu

graph, subgraph, subdpu, subcpu = get_graph(sys.argv[1])

tensor_info = json.loads(subdpu.get_attr('tensor_info'))
for k,v in tensor_info['inputs'].items():
    v['druMode'] = False
tensor_info = json.dumps(tensor_info)
subdpu.set_attr('tensor_info', tensor_info)

graph.serialize(sys.argv[2])
print('New model saved to {}'.format(sys.argv[2]))
