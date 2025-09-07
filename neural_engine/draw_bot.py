import graphviz
from graphviz import nohtml,Digraph

def draw_graph(parent):
    f = graphviz.Digraph(node_attr={'shape': 'record'}, format='svg', graph_attr={'rankdir': 'LR'})
    stack = [parent];
    f.node(str(id(parent)),nohtml(f'<f0> {parent.label} |<f1> Data: {parent.data}|<f2>Grad: {parent.grad}'))
    while len(stack)!=0:
        cur = stack.pop()
        cur_uid = str(id(cur))
        if cur._op != None :
            f.node(name=cur_uid+cur._op, label=cur._op,shape='oval')
            f.edge(cur_uid+cur._op, cur_uid)
        if cur._children is not None:
            for children in cur._children:
                child_uid = str(id(children))
                f.node(child_uid ,nohtml(f'<f0> {children.label} |<f1> Data: {children.data}|<f2>Grad: {children.grad}'))
                if cur._op != None :
                    f.edge(child_uid, cur_uid+cur._op)
                stack.append(children)
    return f


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_bot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(name=str(id(n)), label="{ data %.4f | grad %.4f | label %s}" % (n.data, n.grad, n.label), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot