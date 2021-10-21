"""
You can create any other helper funtions.
Do not modify the given functions
"""

import heapq as hq

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
    
    def neighbours(node,cost):
        visited_t = True
        for i in range(len(cost[node])-1, 0, -1):
            if i!= node and cost[node][i] != -1:
                yield i
    
    path = []
    extra = []
    visited_set = set();
    visited = []  
    priority_q = []
    hq.heappush(priority_q, (heuristic[start_point], [start_point]))
    while(len(priority_q) > 0):
        total_cost, cur_path = hq.heappop(priority_q)
        cur = cur_path[-1]
        total_cost = total_cost - heuristic[cur]
        extra.append(1)
        visited_set.add(1)
        if cur in goals:
            path = cur_path
            break
        if cur not in visited:
            visited.append(cur)
            for i in neighbours(cur, cost):
                if i not in visited:
                    hq.heappush(priority_q, (cost[cur][i] + heuristic[i] + total_cost, cur_path + [i]))
        extra.pop()
    return path


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
    

    
