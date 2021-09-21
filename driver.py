import numpy as np
import cv2
import os
import serial
import heapq
import math
import time
import timeit
import warnings
from selenium import webdriver
from object_detector import *

class bot(object):
    def __init__(self, ID, position, rotation):

        self.id = ID
        self.position = position
        self.rotation = rotation

        self.right_enable = 0
        self.left_enable = 0

        self.orientation = 'SOUTH'
        self.route_path = []

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
        self.rows = 20
        self.cols = 20

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

def draw_path(frame, bot, sx, sy, ex, ey):

    path = bot.route_path
    length = len(path)-1

    cv2.circle(frame,((sy*50)-25,(sx*50)-25),8,(0,165,255),-1)

    for i in range(0,length):
        y1,x1 = path[i]
        y2,x2 = path[i+1]
        cv2.line(frame,((y1*50)-25,(x1*50)-25),((y2*50)-25,(x2*50)-25),(0,165,255),3)

    cv2.circle(frame,((ey*50)-25,(ex*50)-25),8,(0,165,255),-1)

    cv2.imwrite('/home/vishwesh/Desktop/Geekbot/out{}.jpg'.format(bot.id),frame)
    cv2.imshow('frame{}'.format(bot.id), frame)

    return length

def get_start_end(frame):

    height = 508
    width = 1016

    y = 20
    for i in range(0,10):
        x = 20
        if(y < height):
            for j in range (0,20):
                if(x < width):

                    roi = frame.copy()
                    roi = roi[y - 5:y + 5, x - 5:x + 5, :]
                    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
                    hue,sat,val,ret = cv2.mean(hsv)

                    if(hue==0 and sat==255 and val==255):
                        start_x = i
                        start_y = j

                    elif(hue==60 and sat==255 and val==255):
                        end_x = i
                        end_y = j

                    elif (val==0):
                        wall_map.append((i,j))

                    x = x + 50
        y = y + 50

    return (start_x, start_y, end_x, end_y)


def get_perspective_image(frame):

    height = 508
    width = 1016

    t_val = 160
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower = np.array([0, 0, 0])
    upper = np.array([t_val, t_val, t_val])
    mask = cv2.inRange(frame, lower, upper)

    ret,thresh = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    #cv2.polylines(frame, [approx], True, (0,140,255), 3)

    return (dst)

def get_distance(x1,y1,x2,y2):

    distance = math.hypot(x2 - x1, y2 - y1)
    return distance


def communicate(bot):
        for i in range(len(bot.route_path)):
            time.sleep(1)
            if bot.route_path[i] == (0,0):
                pay()

            elif i+1 < len(bot.route_path):
                if (bot.route_path[i+1][0] - bot.route_path[i][0], bot.route_path[i+1][1] - bot.route_path[i][1]) == (0,1):
                    if orientation == 'SOUTH':
                        forward()

                    elif orientation == 'NORTH':
                        back()
                        forward()

                    elif orientation == 'WEST':
                        left()
                        forward()

                    elif orientation == 'EAST':
                        right()
                        forward()

                    orientation = 'SOUTH'

                elif (bot.route_path[i+1][0] - bot.route_path[i][0], bot.route_path[i+1][1] - bot.route_path[i][1]) == (0,-1):
                    if orientation == 'NORTH':
                        forward()

                    elif orientation == 'SOUTH':
                        back()
                        forward()

                    elif orientation == 'WEST':
                        right()
                        forward()

                    elif orientation == 'EAST':
                        left()
                        forward()

                    orientation = 'NORTH'

                elif (bot.route_path[i+1][0] - bot.route_path[i][0], bot.route_path[i+1][1] - bot.route_path[i][1]) == (1,0):
                    if orientation == 'EAST':
                        forward()

                    elif orientation == 'SOUTH':
                        left()
                        forward()

                    elif orientation == 'WEST':
                        back()
                        forward()

                    elif orientation == 'NORTH':
                        right()
                        forward()

                    orientation = 'EAST'

                elif (bot.route_path[i+1][0] - bot.route_path[i][0], bot.route_path[i+1][1] - bot.route_path[i][1]) == (-1,0):
                    if orientation == 'WEST':
                        forward()

                    elif orientation == 'SOUTH':
                        right()
                        forward()

                    elif orientation == 'NORTH':
                        left()
                        forward()

                    elif orientation == 'EAST':
                        back()
                        forward()

                    orientation = 'WEST'

def forward():
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/forward1")
    driver.close()

def back():
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/back1")
    driver.close()

def right():
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/right1")
    driver.close()

def left():
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/left1")
    driver.close()

def pay():
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/pay1")
    driver.close()

def pid():
    pass

def main():
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

    cap = cv2.VideoCapture(0)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    cap.set(3, 1920)
    cap.set(4, 1080)

    out = cv2.VideoWriter('System Tracking.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (1920,1080))

    while(True):
        ret, frame = cap.read()

        if ret == True:

        img = frame.copy()

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 2)

        def wrap_angle(angle):
            # Convert an angle to 0 to 2*pi range.
            new_angle = np.arctan2(np.sin(angle), np.cos(angle))
            if new_angle < 0:
                new_angle = abs(new_angle) + 2 * (np.pi - abs(new_angle))
            return new_angle

        markers = []
        # red, blue, yellow, purple (TL, TR, BR, BL)
        corner_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        for marker in int_corners:
            marker = np.squeeze(marker)
            markers.append(marker)

            for xy_corner, corner_color in zip(marker, corner_colors):
                cv2.circle(img, (xy_corner[0], xy_corner[1]), 8, corner_color, -1)

        marker_locations = []
        marker_angles = []
        # red, blue, yellow, purple (TL, TR, BR, BL) or (0, 1, 2, 3).
        for marker in markers:
            angles = np.array([
                np.arctan2(marker[0][1] - marker[3][1], marker[0][0] - marker[3][0]),
                np.arctan2(marker[1][1] - marker[2][1], marker[1][0] - marker[2][0])])

                # Can use these two parallel lines also, but fails sometimes.
                # np.fmod(np.arctan2(marker[0][1] - marker[1][1], marker[0][0] - marker[1][0]) - np.pi / 2, np.pi),
                # np.fmod(np.arctan2(marker[3][1] - marker[2][1], marker[3][0] - marker[2][0]) - np.pi / 2, np.pi)])

            angle = np.degrees(np.mean(angles))
            marker_locations.append(np.mean(marker, axis=0).astype('int'))
            marker_angles.append(angle)


        for pos in marker_locations:
            cv2.circle(img, (pos[0], pos[1]), 8, (0, 255, 0), -1)

        cv2.imshow('frame',img)
        out.write(img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

main()