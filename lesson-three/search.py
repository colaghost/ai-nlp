import networkx as nx
from pylab import show

import matplotlib
print(matplotlib.__path__)

class Node(object):
    def __init__(self, curr, parent, g, lines):
        self.curr = curr
        self.parent = parent
        self.g = g
        self.lines = set(lines)
    def set_parent(self, parent):
        self.parent = parent
    def set_g(self, g):
        self.g = g
    def __hash__(self):
        return hash(self.curr)
    def __eq__(self, other):
        return self.curr == other.curr

class BJStation():
    def __init__(self, path):
        self.station_line = {}
        self.bj_station = {}
        self.load(path)
        bj_station_graph = nx.Graph(self.bj_station)
        nx.draw(bj_station_graph, with_labels=True, node_size=2)
        #show()

    def load(self, path):
        with open(path) as bj_station_file:
            for line in bj_station_file:
                line = line.strip()
                fields = line.split()
                if len(fields) < 2:
                    continue
                stations = fields[1].split(';')
                if len(stations) == 0:
                    continue
                elif len(stations) == 1:
                    self.station_line.setdefault(stations[0], set())
                    self.station_line[stations[0]].add(fields[0])
                    continue
                for i in range(len(stations)):
                    if i == 0:
                        continue
                    self.station_line.setdefault(stations[i - 1], set())
                    self.station_line[stations[i - 1]].add(fields[0])
                    self.station_line.setdefault(stations[i], set())
                    self.station_line[stations[i]].add(fields[0])

                    self.bj_station.setdefault(stations[i - 1], set())
                    self.bj_station[stations[i - 1]].add(stations[i])
                    self.bj_station.setdefault(stations[i], set())
                    self.bj_station[stations[i]].add(stations[i - 1])
    def cal_line_change_num(self, curr_station, next_station):
        next_station_lines = self.station_line[next_station.curr]
        if len(curr_station.lines & next_station_lines) == 0:
            return curr_station.g + 1
        return curr_station.g
    def cal_line_station_num(self, curr_station, next_station):
        return curr_station.g + 1
    def cal_line_comprehensive_num(self, curr_station, next_station):
        curr_station_g = curr_station.g
        return curr_station_g + (self.cal_line_change_num(curr_station, next_station) - curr_station_g) + (self.cal_line_station_num(curr_station, next_station) - curr_station_g)
    def search_destion(self, src, dest, cal_g_strategy_func):
        close_list = set()
        open_list = set([src])
        station_dict = {src:Node(src, "", 0, self.station_line[src])}
        while len(open_list) > 0:
            curr_station = min(open_list, key=lambda node: station_dict[node].g)
            open_list.remove(curr_station)
            close_list.add(curr_station)
            for next_station in self.bj_station[curr_station]:
                if next_station in close_list:
                    continue
                curr_station_node = station_dict[curr_station]
                if next_station in open_list:
                    next_station_node = station_dict[next_station]
                    cal_g = cal_g_strategy_func(curr_station_node, next_station_node)
                    if cal_g < next_station_node.g:
                        next_station_node.g = cal_g
                        next_station_node.parent = curr_station
                        next_station_node.lines = set(curr_station_node.lines)

                else:
                    lines = set(curr_station_node.lines & self.station_line[next_station])
                    if len(lines) == 0:
                        lines = set(self.station_line[next_station])
                    next_station_node = Node(next_station, curr_station, 0, lines)
                    #cal_g = self.cal_line_change_num(curr_station_node, next_station_node)
                    cal_g = cal_g_strategy_func(curr_station_node, next_station_node)
                    next_station_node.g = cal_g
                    station_dict[next_station] = next_station_node
                    open_list.add(next_station)

        curr_line = set()

        route = []
        curr = dest
        while curr in close_list:
            route.append('{}({})'.format(curr, ','.join(self.station_line[curr])))
            curr = station_dict[curr].parent
        route.reverse()
        for station in route:
            lines = self.station_line[station.split('(')[0]]
            if len(curr_line) == 0:
                curr_line = set(lines)
            else:
                if len(lines & curr_line) == 0:
                    print(curr_line.pop())
                    curr_line = lines
                else:
                    curr_line = set(lines & curr_line)
        if len(curr_line) > 0:
            print(curr_line.pop())
        print(len(route))
        print('->'.join(route))

    def search_min_change_num_destion(self, src, dst):
        return self.search_destion(src, dst, self.cal_line_change_num)

    def search_min_station_num_destion(self, src, dst):
        return self.search_destion(src, dst, self.cal_line_station_num)

    def search_comprehensive_num_destion(self, src, dst):
        return self.search_destion(src, dst, self.cal_line_comprehensive_num)


bj_station = BJStation('/home/parallels/dev/ai-nlp/lesson-three/bj_station')
bj_station.search_comprehensive_num_destion('八角游乐园站', '西四站')
bj_station.search_comprehensive_num_destion('苹果园站', '次渠站')
bj_station.search_min_change_num_destion('西单站', '西直门站')
bj_station.search_min_station_num_destion('西单站', '西直门站')

