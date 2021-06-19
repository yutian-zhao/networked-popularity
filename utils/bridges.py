## Author: Yutian Zhao
## This script implements the algorithm in Firmani, D., Georgiadis, L., Italiano, G. F., Laura, L., & Santaroni, F. (2016). 
## Strong  articulation  points  and  strong  bridges  in  large  scale graphs. Algorithmica, 74(3), 1123-1147. 
## to find strong bridges in a strongly connected graph.
## Main function: lost_nodes, strong_edges
## Reference:
## https://cp-algorithms.com/graph/bridge-searching.html
## https://www.cs.yale.edu/homes/aspnes/pinewiki/DepthFirstSearch.html

from collections import defaultdict
import networkx as nx

def dfs_edges(v, graph):
    # find tree_edges, back_edges, forward_edges, cross_edges in the graph
    # v: starting node
    # graph: a strongly connected graph.

    start = defaultdict(int)
    end = defaultdict(int)
    forward_edges = defaultdict(list)
    back_edges = defaultdict(list)
    cross_edges = defaultdict(list)
    tree_edges = []
    t = 0
        
    def dfs(v, graph):
        nonlocal t
        stack = []
        stack.append((v, None)) # (v, parent)

        while (len(stack)):
            t += 1
            v, parent = stack[-1]
            if start[v] > 0 and end[v] == 0:
                end[v] = t
                stack.pop()
            elif start[v] > 0 and end[v] > 0:
                forward_edges[v].append(parent)
                stack.pop()
            else:
                start[v] = t
                if parent is not None:
                    tree_edges.append((parent, v))
#                 for n in sorted(graph.successors(v), reverse=True):
                for n in graph.successors(v):
                    if start[n] == 0:  # not visited
                        stack.append((n, v))
                    elif end[n] == 0:
                        back_edges[n].append(v)
                    elif start[v] < start[n]:
                        forward_edges[n].append(v)
                    else:
                        cross_edges[n].append(v)
            
    dfs(v, graph)
    
    return tree_edges, back_edges, forward_edges, cross_edges

def dominator_tree(v, graph):
    # find dominate tree in the graph
    # v: starting node
    # graph: a strongly connected graph.
    dt = nx.DiGraph()
    idoms = nx.immediate_dominators(graph, v)
    for k, v in idoms.items():
        if k != v:
            dt.add_edge(v, k)
    return dt

def dfs_order(v, graph):
    # record the first and last traverse time in a depth first search for each node
    # v: starting node
    # graph: a strongly connected graph.

    start = defaultdict(int)
    end = defaultdict(int)
    t = 0
        
    def dfs(v, graph):
        nonlocal t
        stack = []
        stack.append((v, None)) # (v, parent)

        while (len(stack)):
            t += 1
            v, parent = stack[-1]
            if start[v] > 0 and end[v] == 0:
                end[v] = t
                stack.pop()
            elif start[v] > 0 and end[v] > 0:
                stack.pop()
            else:
                start[v] = t
#                 for n in sorted(graph.successors(v), reverse=True):
                for n in graph.successors(v):
                    if start[n] == 0:  # not visited
                        stack.append((n, v))
        
    dfs(v, graph)
    
    return start, end

def edge_dominators(v, graph): 
    # find edge dominators
    # v: starting node
    # graph: a strongly connected graph.   
    edge_dominator = set()
    tree_edges, back_edges, forward_edges, cross_edges = dfs_edges(v, graph)
    dt = dominator_tree(v, graph)
    start, end = dfs_order(v, dt)

    for e in tree_edges:
        if len(forward_edges[e[1]])==0 and len(cross_edges[e[1]])==0:
            flag = True
            for i in back_edges[e[1]]:  # for every back edge, e_1 dominate i
                if not ((start[e[1]] < start[i]) and (end[i] < end[e[1]])):
                    flag = False
            if flag:
                edge_dominator.add(e)

    return edge_dominator

def strong_edges(v, graph):
    # find all strong bridges in the graph
    # v: starting node
    # graph: a strongly connected graph.
    de = edge_dominators(v, graph)
    for v, u in edge_dominators(v, graph.reverse()):
        de.add((u, v))
    return de

def lost_nodes(bridges, subgraph, graph, mode):
    ## collect lost nodes if a (strong) bridge is deleted in a (strongly) weakly connected component.
    ## bridges: all bridges which will be deleted with replacement.
    ## subgraph: (strongly) weakly connected component.
    ## graph: the whole graph containing subgraph.
    ## mode: 'strong' or 'weak'.
    sg = subgraph.copy()

    nodes_list = []
    for se in bridges:
        nodes = []
        
        if mode=='weak':
            # in weakly connected graph, an bridge may be stored in (s, t) or (t, s)
            if sg.has_edge(*se):
                src = se[0]
                tgt = se[1]
            else:
                src = se[1]
                tgt = se[0]
            sg.remove_edge(src, tgt)                       
            components = sorted(nx.weakly_connected_components(sg), key=len)
            sg.add_edge(src, tgt)

        else:
            sg.remove_edge(*se)
            components = sorted(nx.strongly_connected_components(sg), key=len)
            sg.add_edge(*se)

        count = 0
        view_sum = 0
        indegree_sum = 0
        for c in components[:-1]:
            for v in c:
                count += 1
                nodes.append(v)

        # for debugging
        if count == 0:
            if not (sg.has_edge(*se) and sg.has_edge(se[1], se[0])):
                        print("graph: ", i, " has ", [len(component) for component in components], " components without ", se)
                        return None
        else:
            nodes_list.append(nodes)        
    
    return nodes_list