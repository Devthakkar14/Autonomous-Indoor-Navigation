import heapq
from threading import *
import time
import math

#all CoOrs in wall map are blocked for the bot
#wall_map = [(2,2), (3,2), (2,3), (3,3), (2,6), (3,6), (2,7), (3,7), (2,10), (3,10), (2,11), (3,11), (6,2), (7,2), (6,3), (7,3), (6,6), (7,6), (6,7), (7,7), (6,10), (7,10), (6,11), (7,11), (10,2), (11,2), (10,3), (11,3), (10,6), (11,6), (10,7), (11,7), (10,10), (11,10), (10,11), (11,11)]
wall_map = []
route = []

class Node:

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __init__(node, x, y, space):

        node.x = x
        node.y = y

        node.space = space
        node.parent = None

        node.f = 0.0
        node.g = 0.0
        node.h = 0.0
        node.sem = Semaphore()

class path_algorithm:

    def __init__(self):

        self.open_list  = []
        self.close_list = set()

        heapq.heapify(self.open_list)

        self.nodes = []
        self.lockedNodes = []
        self.rows = 14
        self.cols = 14

    def grid_map(self, sx, sy, ex, ey):

        for i in range(0,self.rows):
            for j in range(0,self.cols):

                if (i,j) in wall_map:
                    space = 0
                else:
                    space = 1
                self.nodes.append(Node(i,j,space))

        self.start = self.get_pos(sx,sy)
        self.end   = self.get_pos(ex,ey)

    def get_pos(self, x, y):

        pos = self.nodes[(x*self.rows)+y]
        return pos

    def get_adjacent(self, node):

        adj_nodes = []

        if (node.x < self.cols-1 and self.get_pos(node.x+1, node.y).sem.acquire(blocking=False)):
            adj_nodes.append(self.get_pos(node.x+1, node.y))
            self.lockedNodes.append(self.get_pos(node.x+1, node.y))
        if (node.y > 0 and self.get_pos(node.x, node.y-1).sem.acquire(blocking=False)):
            adj_nodes.append(self.get_pos(node.x, node.y-1))
            self.lockedNodes.append(self.get_pos(node.x, node.y-1))
        if (node.x > 0 and self.get_pos(node.x-1, node.y).sem.acquire(blocking=False)):
            adj_nodes.append(self.get_pos(node.x-1, node.y))
            self.lockedNodes.append(self.get_pos(node.x-1, node.y))
        if (node.y < self.rows-1 and self.get_pos(node.x, node.y+1).sem.acquire(blocking=False)):
            adj_nodes.append(self.get_pos(node.x, node.y+1))
            self.lockedNodes.append(self.get_pos(node.x, node.y+1))

        return adj_nodes

    def get_h(self, node):

        h_factor = 20
        dx = abs(node.x - self.end.x)
        dy = abs(node.y - self.end.y)
        h  = h_factor * (dx + dy)

        return h

    def update_values(self, adj, node):

        adj.g = node.g + 50
        adj.h = self.get_h(adj)
        adj.f = adj.g + adj.h
        adj.parent = node

    def path_list(self, route_path):

        node = self.end
        while(node.parent is not self.start):
            node = node.parent
            route_path.append((node.y+1,node.x+1))

    def path_detect(self, route_path):

        heapq.heappush(self.open_list, (self.start.f, self.start))

        while(len(self.open_list)):

            f,node = heapq.heappop(self.open_list)
            self.close_list.add(node)

            if node is self.end:
                self.path_list(route_path)
                break

            adj_list = self.get_adjacent(node)

            for adj in adj_list:
                if(adj.space and (adj not in self.close_list)):
                    if((adj.f, adj) in self.open_list):
                        if(adj.g > (node.g + 50)):
                            self.update_values(adj, node)
                    else:
                        self.update_values(adj, node)
                        heapq.heappush(self.open_list, (adj.f, adj))

    def releaseSem(self):
        for i in self.lockedNodes:
            i.sem.release

if __name__ == '__main__':
    route_path = []
    start_x = 0;
    start_y = 0;
    end_x = 4;
    end_y = 4;
    route_path.append((end_y+1, end_x+1))
    path = path_algorithm()
    path.grid_map(start_x, start_y, end_x, end_y)
    path.path_detect(route_path)
    route_path.append((start_y+1, start_x+1))
    route_path.reverse()
    print(route_path)
    path.releaseSem