graph = {
    'A': 'B C',
    'B': 'D E',
    'C': 'A',
    'D': 'B',
    'E': 'B F',
    'F': 'E'
}
for k in graph.keys():
    graph[k] = graph[k].split()


def BreadthFirstEnqueueFunc(next_search_points, search_queue):
    return search_queue + next_search_points


def DepthFirstEnqueueFunc(next_search_points, search_queue):
    return next_search_points + search_queue


def Search(graph, src, dst, EnqueueFunc):
    try_routes = [src]
    filters = {}
    while len(try_routes) > 0:
        point = try_routes.pop(0)
        if point in filters or point not in graph:
            continue
        filters[point] = 1
        print(point)
        if point == dst:
            return True
        try_routes = EnqueueFunc(graph[point], try_routes)
    return False


def BFS(graph, src, dst):
    return Search(graph, src, dst, BreadthFirstEnqueueFunc)


def DFS(graph, src, dst):
    return Search(graph, src, dst, DepthFirstEnqueueFunc)


DFS(graph, 'A', 'C')

print('xxxxxxxxxxxxxxxxxxxxxxxxxxx')

BFS(graph, 'A', 'C')

AirLineGraph = {
    "SZ": "BJ",
    "BJ": "SH GZ  HB",
    "SH": "BJ",
    "GZ": "BJ",
    "HB": "BJ WH NJ",
    "WH": "HB",
    "NJ": "HB HK",
    "HK": "NJ"
}

for key in AirLineGraph:
    AirLineGraph[key] = AirLineGraph[key].split()

import networkx
from pylab import show

def DrawGraph(graph):
    graph_ = networkx.Graph(graph)
    networkx.draw(graph_, with_labels=True)
    show()

def SearchAirLine(graph, src, dst):
    filters = {}
    try_routes = [[src]]
    while len(try_routes) > 0:
        curr_route = try_routes.pop(0)
        point = curr_route[-1]
        if point in filters or point not in graph:
            continue
        filters[point] = 1
        if point == dst:
            return curr_route
        for next_point in graph[point]:
            try_routes.append(curr_route + [next_point])

    return []


route = SearchAirLine(AirLineGraph, "SZ", "HK")
print('->'.join(route))
