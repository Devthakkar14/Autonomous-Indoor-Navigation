import numpy as np
import cv2
import os
import serial
import heapq
import math
import time
import timeit
import warnings
from object_detector import *
from threading import *

wall_map = [(3,3), (3,4), (4,3), (4,4), (7,3), (7,4), (8,3), (8,4), (11,3), (11,4), (12,3), (12,4), (3,7), (3,8), (4,7), (4,8), (7,7), (7,8), (8,7), (8,8), (11,7), (11,8), (12,7), (12,8), (3,11), (3,12), (4,11), (4,12), (7,11), (7,12), (8,11), (8,12), (11,11), (11,12), (12,11), (12,12),(11,15),(12,15),(13,15),(0,15),(1,15),(2,15),(3,15),(4,15),(6,15),(7,15),(8,15),(9,15),(14,15),(15,15)]
bot1 = []
bot2 = []
sem = Semaphore()

class Node(object):

    def __lt__(self, other):
        self.f < other.f

    def __le__(self, other):
        self.f <= other.f

    def __init__(node, x, y, space):

        node.x = x
        node.y = y

        node.space = space
        node.parent = None

        node.f = 0.0
        node.g = 0.0
        node.h = 0.0

class path_algorithm(object):
    def __init__(self):

        self.open_list  = []
        self.close_list = set()

        heapq.heapify(self.open_list)

        self.nodes = []
        self.rows = 15
        self.cols = 15

    def grid_map(self, sx, sy, ex, ey):

        for i in range(0,self.rows):
            for j in range(0,self.cols):

                if (i+1,j+1) in wall_map:
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

        if (node.x < self.cols-1):
            adj_nodes.append(self.get_pos(node.x+1, node.y))
        if (node.y > 0):
            adj_nodes.append(self.get_pos(node.x, node.y-1))
        if (node.x > 0):
            adj_nodes.append(self.get_pos(node.x-1, node.y))
        if (node.y < self.rows-1):
            adj_nodes.append(self.get_pos(node.x, node.y+1))

        return adj_nodes

    def get_h(self, node):
        h_factor = -20
        dx = abs(node.x - self.end.x)
        dy = abs(node.y - self.end.y)
        h = abs((node.x*self.end.x) - (self.end.y*node.y))*0.001

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

def draw_path(path, ex, ey, img, imgDest, color):

    #Path length
    length = len(path)-1

    opacity = 0.2
    cv2.addWeighted(img, opacity, img, 1 - opacity, 0, img)

    #Draw the path lines
    for i in range(0,length):
        y1,x1 = path[i]
        y2,x2 = path[i+1]
        x1 += 1
        x2 += 1
        y1 += 1
        y2 += 1
        cv2.line(img,((y1*53)-30,(x1*55)-18),((y2*53)-30,(x2*55)-18),color,3)

    cv2.imwrite(imgDest,img)

    return length

def get_perspective_image(frame):

    height = 900
    width = 890

    t_val = 160
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower = np.array([0, 0, 0]) #black color mask
    upper = np.array([t_val, t_val, t_val])
    mask = cv2.inRange(frame, lower, upper)

    ret,thresh = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,contours,-1,(0,255,0),3)
    biggest = 0
    max_area = 0
    min_size = thresh.size/4
    index1 = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 10000:
            peri = cv2.arcLength(i,True)

        if area > max_area:
            biggest = index1
            max_area = area
        index1 = index1 + 1
    approx = cv2.approxPolyDP(contours[biggest],0.05*peri,True)

    x1 = approx[0][0][0]
    y1 = approx[0][0][1]
    x2 = approx[1][0][0]
    y2 = approx[1][0][1]
    x3 = approx[3][0][0]
    y3 = approx[3][0][1]
    x4 = approx[2][0][0]
    y4 = approx[2][0][1]

    raw_points = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    min_dist = 1000000
    for i in raw_points:
        x,y = i
        dist = get_distance(x,y,0,0)
        if dist < min_dist:
            min_dist = dist
            X1,Y1 = x,y

    min_dist = 1000000
    for i in raw_points:
        x,y = i
        dist = get_distance(x,y,0,height)
        if dist < min_dist:
            min_dist = dist
            X2,Y2 = x,y

    min_dist = 1000000
    for i in raw_points:
        x,y = i
        dist = get_distance(x,y,width,0)
        if dist < min_dist:
            min_dist = dist
            X3,Y3 = x,y

    min_dist = 1000000
    for i in raw_points:
        x,y = i
        dist = get_distance(x,y,width,height)
        if dist < min_dist:
            min_dist = dist
            X4,Y4 = x,y

    pts1 = np.float32([[X1,Y1],[X2,Y2],[X3,Y3],[X4,Y4]])
    pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])

    persM = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(frame,persM,(width,height))
    corner_points = [[x4,y4],[x2,y2],[x3,y3],[x1,y1]]

    x_coordinate = 7
    y_coordinate = 2

    X1 = 0+(x_coordinate+1)*53-5
    X2 = X1+53-5
    Y1 = 0+(y_coordinate+1)*57-5
    Y2 = Y1+57-5

    #img = dst[Y1:Y2, X1:X2]
    #print(check_color(img))
    #drawing the biggest polyline
    #cv2.polylines(frame, [approx], True, (0,140,255), 3)
    #cv2.rectangle(dst, (X1,Y1), (X2,Y2), (0,255,0), 2)
    #cv2.circle(dst,(50, 50),4,(0,0,255),2)
    '''for i in range(14):
        for j in range(14):
            cv2.circle(dst,(75+i*53, 75+j*57),4,(0,0,255),2)'''
    cv2.imwrite('/home/vishwesh/GeekBot/out.jpg',dst)

    return (dst)

'''
def check_color(img):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hue, sat, val, ret = cv2.mean(hsv)
    print(hue, sat, val)

    tol = 30

    if 114.7-tol <= hue <= 114.7+tol and 127.67-tol <= sat <= 127.67+tol and 95.63-tol <= val <= 95.63+tol :
        return "BLUE"
    if 20.5-tol <= hue <= 20.5+tol and 148.9-tol <= sat <= 148.9+tol and 218.9-tol <= val <= 218.9+tol :
        return "LI-ORANGE"
    if 49.2-tol <= hue <= 49.2+tol and 95.5-tol <= sat <= 95.5+tol and 208-tol <= val <= 208+tol :
        return "LI-GREEN"
    if 59.78-tol <= hue <= 59.78+tol and 169.60-tol <= sat <= 169.60+tol and 182.09-tol <= val <= 182.09+tol :
        return "RED"
    if 166.94-tol <= hue <= 166.94+tol and 123.7-tol <= sat <= 123.7+tol and 251.45-tol <= val <= 251.45+tol :
        return "PINK"
    if 15.24-tol <= hue <= 15.24+tol and 203.9-tol <= sat <= 203.9+tol and 194.49-tol <= val <= 194.49+tol :
        return "DA-ORANGE"
    if 75.50-tol <= hue <= 75.50+tol and 116.02-tol <= sat <= 116.02+tol and 149.03-tol <= val <= 149.03+tol :
        return "DA-GREEN"
    if 107.78-tol <= hue <= 107.78+tol and 138.74-tol <= sat <= 138.74+tol and 218.83-tol <= val <= 218.83+tol :
        return "LI-BLUE"
    else:
        return "WHITE"'''

def get_distance(x1,y1,x2,y2):

    distance = math.hypot(x2 - x1, y2 - y1)
    return distance

def thread_handler_1(frame):
    global wall_map, bot1
    route_path = []
    start_x = 15;
    start_y = 5;
    end_x = 4;
    end_y = 5;

    sem.acquire()
    while((end_y, end_x) in wall_map):
        pass
    route_path.append((end_x, end_y))
    path = path_algorithm()
    path.grid_map(start_y-1, start_x-1, end_y-1, end_x-1)
    path.path_detect(route_path)
    route_path.append((start_x, start_y))
    route_path.reverse()
    path_color = (255,0,0)
    draw_path(route_path, end_x, end_y, frame, '/home/vishwesh/GeekBot/view1.jpg', path_color)
    print(route_path)
    obs = np.asarray([sublist for sublist in route_path])
    obs = obs.T
    bot1 = list(zip(obs[1],obs[0]))
    wall_map += bot1
    wall_map = list(set(wall_map) - set(bot2))
    sem.release()

def thread_handler_2(frame):
    global wall_map, bot2
    route_path = []
    start_x = 15;
    start_y = 10;
    end_x = 6;
    end_y = 5;

    sem.acquire()
    while((end_y, end_x) in wall_map):
        pass
    route_path.append((end_x, end_y))
    path = path_algorithm()
    path.grid_map(start_y-1, start_x-1, end_y-1, end_x-1)
    path.path_detect(route_path)
    route_path.append((start_x, start_y))
    route_path.reverse()
    path_color = (0,0,255)
    draw_path(route_path, end_x, end_y, frame, '/home/vishwesh/GeekBot/view2.jpg', path_color)
    print(route_path)
    obs = np.asarray([sublist for sublist in route_path])
    obs = obs.T
    bot2 = list(zip(obs[1],obs[0]))
    wall_map += bot2
    wall_map = list(set(wall_map) - set(bot1))
    sem.release()

if __name__ == '__main__':
    frame = cv2.imread('/home/vishwesh/GeekBot/2.jpg')
    alpha = 1.2 # Contrast control (1.0-3.0)
    beta = 5 # Brightness control (0-100)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    cv2.imwrite('/home/vishwesh/GeekBot/test.jpg',frame)
    frame = get_perspective_image(frame)

    t1 = Thread(target=thread_handler_1, args=(frame,))
    t2 = Thread(target=thread_handler_2, args=(frame,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()