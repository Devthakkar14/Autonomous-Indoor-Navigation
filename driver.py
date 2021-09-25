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
        self.error = 0

        self.P = 0
        self.I = 0
        self.D = 0

        self.Kp = 0
        self.Ki = 0
        self.Kd = 0

        self.orientation = 'SOUTH'
        self.next_orientation = 'SOUTH'
        self.next = 'NONE'
        self.target = (0,0)
        self.route_path = []

        self.complete = False

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

def get_position(virtual):
    return real

def target(bot):
    for i in range(len(bot.route_path)):
        if bot.route_path[i] == (0,0):
            bot.next = 'PAY'
            bot.route_path = bot.route_path[i+1:]
            break

        elif i+1 == len(bot.route_path):
            bot.route_path = []
            bot.next = 'END'

        elif i+1 < len(bot.route_path):
            x2 = bot.route_path[i+1][0]
            x1 = bot.route_path[i][0]
            y2 = bot.route_path[i+1][1]
            y1 = bot.route_path[i][1]
            if (x2-x1, y2-y1) == (0,1):
                if bot.orientation == 'SOUTH':
                    bot.target = (x2,y2)
                    continue

                elif bot.orientation == 'NORTH':
                    bot.next = 'BACK'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'SOUTH'
                    break

                elif bot.orientation == 'WEST':
                    bot.next = 'LEFT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'SOUTH'
                    break

                elif bot.orientation == 'EAST':
                    bot.next = 'RIGHT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'SOUTH'
                    break

            elif (x2-x1, y2-y1) == (0,-1):
                if bot.orientation == 'NORTH':
                    bot.target = (x2,y2)
                    continue

                elif bot.orientation == 'SOUTH':
                    bot.next = 'BACK'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'NORTH'
                    break

                elif bot.orientation == 'WEST':
                    bot.next = 'RIGHT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'NORTH'
                    break

                elif bot.orientation == 'EAST':
                    bot.next = 'LEFT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'NORTH'
                    break

            elif (x2-x1, y2-y1) == (1,0):
                if bot.orientation == 'EAST':
                    bot.target = (x2,y2)
                    continue

                elif bot.orientation == 'SOUTH':
                    bot.next = 'LEFT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'EAST'
                    break

                elif bot.orientation == 'WEST':
                    bot.next = 'BACK'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'EAST'
                    break

                elif bot.orientation == 'NORTH':
                    bot.next = 'RIGHT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'EAST'
                    break

            elif (x2-x1, y2-y1) == (-1,0):
                if bot.orientation == 'WEST':
                    bot.target = (x2,y2)
                    continue

                elif bot.orientation == 'SOUTH':
                    bot.next = 'RIGHT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'WEST'
                    break

                elif bot.orientation == 'NORTH':
                    bot.next = 'LEFT'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'WEST'
                    break

                elif bot.orientation == 'EAST':
                    bot.next = 'BACK'
                    bot.route_path = bot.route_path[i:]
                    bot.next_orientation = 'WEST'
                    break
    return bot

def right(bot):
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/right")
    driver.close()

def left(bot):
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/left")
    driver.close()

def back(bot):
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/back")
    driver.close()

def pay(bot):
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/pay")
    driver.close()

def wrap_angle(angle):
    # Convert an angle to 0 to 2*pi range.
    new_angle = np.arctan2(np.sin(angle), np.cos(angle))
    if new_angle < 0:
        new_angle = abs(new_angle) + 2 * (np.pi - abs(new_angle))
    return new_angle

def final_path(bot, img):

    grn = (0,255,0)
    red = (0,0,255)
    blk = (0,0,0)
    blu = (255,0,0)

    img2[0:400,362:412] = blk
    img2[350:400,0:406] = blk
    img2[0:400,606:656] = blk
    img2[350:400,606:] = blk

    img2[0:50,412:462] = red
    img2[400:450,56:106] = grn

    cv2.imwrite('/home/vishwesh/Desktop/Geekbot/mid1.jpg',img)

    start_x,start_y,end_x,end_y = get_start_end(img)
    bot.route_path.append((end_y+1, end_x+1))

    self = path_algorithm()
    self.grid_map(start_x, start_y, end_x, end_y)
    self.path_detect(bot.route_path)

    bot.route_path.append((start_y+1, start_x+1))

    route_path_copy = bot.route_path
    bot.route_path = list(reversed(bot.route_path))

    draw_path(img, bot, end_x+1, end_y+1, start_x+1, start_y+1)

    bot.route_path.append((0,0))
    bot.route_path += route_path_copy[0:]

    print('Route Path {}:'.format(bot.id), bot.route_path)
    print()
    return bot

def pid(bot, current_position):
    if bot.orientation == 'SOUTH':
        error = current_position[0] - bot.position[0]
    elif bot.orientation == 'NORTH':
        error = bot.position[0] - current_position[0]
    elif bot.orientation == 'WEST':
        error = bot.position[1] - current_position[1]
    elif bot.orientation == 'EAST':
        error = current_position[1] - bot.position[1]

    bot.P = error
    bot.I += error
    bot.D = error - bot.error
    pid = (bot.Kp*bot.P) + (bot.Ki*bot.I) + (bot.Kd*bot.D)

    bot.right_enable -= pid
    bot.left_enable += pid
    bot.error = error

    return bot

if __name__ == '__main__':

    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

    cap = cv2.VideoCapture(0)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")
        break

    cap.set(3, 1920)
    cap.set(4, 1080)

    bot1 = bot(1, position, rotation)
    bot2 = bot(2, position, rotation)
    bot3 = bot(3, position, rotation)
    bot4 = bot(4, position, rotation)

    out = cv2.VideoWriter('System Tracking.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (1920,1080))

    os.system("nmcli con up GeekNetwork1")

    while (bot1.complete == False):
        ret, frame = cap.read()

        if ret == True:

            img = get_perspective_image(frame)

            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 2)

            markers = []

            corner_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
            for marker in int_corners:
                marker = np.squeeze(marker)
                markers.append(marker)

                for xy_corner, corner_color in zip(marker, corner_colors):
                    cv2.circle(img, (xy_corner[0], xy_corner[1]), 8, corner_color, -1)

            marker_locations = []
            marker_angles = []

            for marker in markers:
                angles = np.array([
                    np.arctan2(marker[0][1] - marker[3][1], marker[0][0] - marker[3][0]),
                    np.arctan2(marker[1][1] - marker[2][1], marker[1][0] - marker[2][0])])

                angle = np.degrees(np.mean(angles))
                angle = wrap_angle(angle)
                marker_locations.append(np.mean(marker, axis=0).astype('int'))
                marker_angles.append(angle)

            for pos in marker_locations:
                cv2.circle(img, (pos[0], pos[1]), 8, (0, 255, 0), -1)

            if bot1.next == 'NONE':
                bot1 = final_path(bot1, img)

            if current_position == get_position(bot1.target) and bot1.next == 'END':
                bot1.complete = True
                break

            elif current_position == get_position(bot1.target):
                if bot1.next == 'RIGHT':
                    right()
                elif bot1.next == 'LEFT':
                    left()
                elif bot1.next == 'BACK':
                    back()
                elif bot1.next == 'PAY':
                    pay()
                bot1.orientation = bot1.next_orientation
                bot1.next = target(bot1)

            bot1 = pid(bot1, position)

            driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
            driver.get("http://192.168.11.6/forward?r="+str(right_enable)+"&l="+str(left_enable))
            driver.close()

            cv2.imshow('frame',img)
            out.write(img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()