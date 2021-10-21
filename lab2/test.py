"""
You can create any other helper funtions.
Do not modify the given functions
"""

import heapq as hq
from functools import cmp_to_key

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    class MyHeap(object):
        def __init__(self, initial=None, key=lambda x:x):
            self.key = key
            self.index = 0
            if initial:
                self._data = [(key(item), i, item) for i, item in enumerate(initial)]
                self.index = len(self._data)
                hq.heapify(self._data)
            else:
                self._data = []

        def push(self, item):
            hq.heappush(self._data, (self.key(item), self.index, item))
            self.index += 1

        def pop(self):
            return hq.heappop(self._data)[2]

        def peak(self):
            return self._data[0][2]

    path = []
    path_list = []
    visited_set = set()
    nodes = [[neighbour_vertex_wt + heuristic[neigh_indx], neighbour_vertex_wt, neigh_indx, start_point] for neigh_indx, neighbour_vertex_wt in enumerate(cost[start_point]) if neighbour_vertex_wt > 0]
    weights_dict = {neigh_indx:neighbour_vertex_wt for neigh_indx, neighbour_vertex_wt in enumerate(cost[start_point]) if neighbour_vertex_wt > 0}
    visited_set.add(start_point)
    hq.heapify(nodes)
    nodes = sorted(nodes, key=cmp_to_key(lambda item1, item2: item1[0] - item2[0]))

    print(nodes)
    print(weights_dict)


    while nodes[0][2] not in goals:
        current_node = nodes[0]
        print("################################################")
        print("nodes")
        print(nodes)
        print("visited_set")
        print(visited_set)
        print("current_node")
        print(current_node)
        path_list.append(nodes[0])
        hq.heappop(nodes)
        if current_node[2] in visited_set:
            continue
        
        neigh_nodes = [[neighbour_vertex_wt + heuristic[neigh_indx], neighbour_vertex_wt, neigh_indx, current_node[2]] for neigh_indx, neighbour_vertex_wt in enumerate(cost[current_node[2]]) if neighbour_vertex_wt > 0]
        neigh_nodes = sorted(neigh_nodes, key=cmp_to_key(lambda item1, item2: item1[0] - item2[0]))
        print("neigh_nodes")
        print(neigh_nodes)
        for neigh_node in neigh_nodes:
            neigh_node_list = neigh_node
            curr_vertex = neigh_node_list[2]
            if curr_vertex in visited_set:
                continue
            if curr_vertex in weights_dict:
                if weights_dict[curr_vertex] > current_node[1] + neigh_node_list[1] :
                    for i in range(len(nodes)):
                        if nodes[i][2] == curr_vertex:
                            weights_dict[curr_vertex] = current_node[1] + neigh_node_list[1]
                            nodes[i][0] = current_node[1] + neigh_node_list[1] + heuristic[curr_vertex]
                            nodes[i][1] = current_node[1] + neigh_node_list[1]
                            nodes[i][3] = current_node[2]
                            break
                    
                    nodes = sorted(nodes, key=cmp_to_key(lambda item1, item2: item1[0] - item2[0]))
            else:
                weights_dict[curr_vertex] = current_node[1] + neigh_node_list[1]
                neigh_node_list[0] = current_node[1] + neigh_node_list[1] + heuristic[curr_vertex]
                neigh_node_list[1] = current_node[1] + neigh_node_list[1]
                neigh_node_list[3] = current_node[2]
                hq.heappush(nodes, neigh_node_list)
                nodes = sorted(nodes, key=cmp_to_key(lambda item1, item2: item1[0] - item2[0]))
        
        
        visited_set.add(current_node[2])

    def find_path_from_list(pq : list) -> list:
        final_l = []
        start_n = nodes[0]
        final_l.insert(0, start_n[2])
        while start_n[3] != start_point:
            final_l.insert(0, start_n[3])
            for nod in path_list:
                if nod[2] == start_n[3]:
                    start_n = nod
                    break
        final_l.insert(0, start_point)
        return final_l

    print("nodes[0]")
    print(nodes[0])
    print("path_list")
    print(find_path_from_list(nodes))

    return find_path_from_list(nodes)


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """

    path = []
    visited_set = set()

    def DFS_Helper(current_vertex, visited_set):
        visited_set.add(current_vertex)
        path.append(current_vertex)

        if current_vertex in goals:
            return True

        for idx, neighbour_vertex_wt in enumerate(cost[current_vertex]):
            if neighbour_vertex_wt > 0:
                if idx not in visited_set:
                    result = DFS_Helper(idx, visited_set)
                    if result:
                        return result

    DFS_Helper(start_point, visited_set)

    return path
    

    
