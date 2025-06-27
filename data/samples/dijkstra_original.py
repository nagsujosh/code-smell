import heapq

def dijkstra(graph, source):
    distance = {node: float('inf') for node in graph}
    distance[source] = 0
    visited = set()
    min_heap = [(0, source)]

    while min_heap:
        curr_dist, curr_node = heapq.heappop(min_heap)
        if curr_node in visited:
            continue
        visited.add(curr_node)

        for neighbor, weight in graph[curr_node]:
            new_dist = curr_dist + weight
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                heapq.heappush(min_heap, (new_dist, neighbor))

    return distance

if __name__ == "__main__":
    g = {
        'A': [('B', 1), ('C', 4)],
        'B': [('C', 2), ('D', 5)],
        'C': [('D', 1)],
        'D': []
    }

    src = 'A'
    shortest = dijkstra(g, src)
    print(f"Shortest distances from {src}:")
    for node in sorted(shortest):
        print(f"{node}: {shortest[node]}")