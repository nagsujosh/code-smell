import heapq

def shortest_path(graph_data, start_node):
    # initialize distances to all nodes
    dist_map = {key: float("inf") for key in graph_data}
    dist_map[start_node] = 0
    visited_set = set()
    pq = [(0, start_node)]

    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited_set:
            continue
        visited_set.add(node)

        for adj, cost in graph_data[node]:
            temp = dist + cost
            if temp < dist_map[adj]:
                dist_map[adj] = temp
                heapq.heappush(pq, (temp, adj))

    return dist_map

if __name__ == "__main__":
    network = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
    }

    origin = 'A'
    result = shortest_path(network, origin)
    print("From", origin)
    for loc in sorted(result):
        print(f"{loc}: {result[loc]}")