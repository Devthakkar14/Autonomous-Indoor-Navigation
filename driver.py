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

wall_map = [(0, 7), (0, 12), (1, 7), (1, 12), (2, 7), (2, 12), (3, 7), (3, 12), (4, 7), (4, 12), (5, 7), (5, 12), (6, 7), (6, 12), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19)]


class bot(object):
    def __init__(self, ID, position, rotation):

        self.id = ID
        self.position = position
        self.rotation = rotation

        self.target_position = (-1,-1)

        self.right_enable = 200
        self.left_enable = 200

        self.error = 0
        self.error_r = 0

        self.P = 0
        self.Kp = 1

        self.Pr = 0
        self.Kpr = 1

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

    return bot

def get_perspective_image(frame):

    height = 500
    width = 1000

    t_val = 160
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower = np.array([0, 0, 0]) #black color mask
    upper = np.array([t_val, t_val, t_val])
    mask = cv2.inRange(frame, lower, upper)

    ret,thresh = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imwrite('/home/vishwesh/Desktop/Geekbot/view1.jpg',thresh)

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

    #drawing the biggest polyline
    #cv2.polylines(frame, [approx], True, (0,140,255), 3)

    cv2.imwrite('/home/vishwesh/Desktop/Geekbot/view.jpg',frame)

    return (dst)

def get_distance(x1,y1,x2,y2):

    distance = math.hypot(x2 - x1, y2 - y1)
    return distance

def get_position(virtual):
    coordinates = { (0, 0) : (-1, -1),
                    (9, 1) : (430,35),
                    (10,1) : (480,35),
                    (11,1) : (525,35),
                    (12,1) : (575,35),
                    (9, 9) : (430,415),
                    (10,10) : (480,450),
                    (11,10) : (525,450),
                    (12,9) : (575,415),
                    (2, 9) : (92,415),
                    (2,10) : (92,450),
                    (19,10) : (910,450),
                    (19,9) : (910,415), }

    return coordinates[virtual]

def within(a, b):

    if a[0] <= (b[0]+27) and a[0] >= (b[0]-27) and a[1] <= (b[1]+27) and a[1] >= (b[1]-27):
        return True

    return False

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
    driver.get("http://192.168.11.6/right?r=190&l=200")
    driver.close()

def left(bot):
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/left?r=220&l=200")
    driver.close()

def back(bot):
    driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
    driver.get("http://192.168.11.6/back?r=190&l=200")
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

def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (230, 230, 230)
    thickness = cv2.FILLED

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + 5
    end_y = pos[1] - txt_size[0][1] - 5

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def final_path(bot, img, start_x, start_y, end_x, end_y):

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
    if bot.orientation == 'NORTH':
        error = current_position[0] - bot.target_position[0]
    elif bot.orientation == 'SOUTH':
        error = bot.target_position[0] - current_position[0]
    elif bot.orientation == 'EAST':
        error = current_position[1] - bot.target_position[1]
    elif bot.orientation == 'WEST':
        error = bot.target_position[1] - current_position[1]

    pid = 0
    bot.P = error
    pid = (bot.Kp*bot.P)

    if pid >= 7 or pid <= -7:
        bot.right_enable += pid
        bot.left_enable -= pid

    print("PID Position: ",pid)

    bot.error = error
    return bot

def pid_r(bot, current_rotation):
    if bot.orientation == 'SOUTH':
        error = current_rotation + 90
    elif bot.orientation == 'NORTH':
        error = current_rotation - 90
    elif bot.orientation == 'WEST':
        error = current_rotation
    elif bot.orientation == 'EAST':
        if current_rotation <= 180 and current_rotation > 0:
            error = current_rotation - 180
        elif current_rotation >= -180 and current_rotation < 0:
            error = current_rotation + 180

    bot.Pr = error
    bot.error_r = error

    pid = (bot.Kpr*bot.Pr)
    bot.right_enable += pid
    bot.left_enable -= pid

    print("PID Rotation: ",pid)

    return bot

if __name__ == '__main__':

    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

    cap = cv2.VideoCapture(0)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    cap.set(3, 1920)
    cap.set(4, 1080)

    start = time.time()

    bot1 = bot(1, (-1,-1), 0)
    bot2 = bot(2, (-1,-1), 0)
    bot3 = bot(3, (-1,-1), 0)
    bot4 = bot(4, (-1,-1), 0)

    out = cv2.VideoWriter('System_Tracking.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 0.5, (1000,500))

    os.system("nmcli con up GeekNetwork1")

    bot1.right_enable = 220
    bot1.left_enable = 200

    while (bot1.complete == False):
        for i in range(7):
            ret, frame = cap.read()
            if i == 6:
                break

        if ret == True:
            alpha = 2 # Contrast control (1.0-3.0)
            beta = 10 # Brightness control (0-100)

            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
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
                marker_locations.append(np.mean(marker, axis=0).astype('int'))
                marker_angles.append(angle)

            for pos in marker_locations:
                cv2.circle(img, (pos[0], pos[1]), 8, (0, 255, 0), -1)

            try:
                ids = ids.flatten()

            except AttributeError:
                print('ArUco marker was not detected')
                continue

            for i in range(len(ids)):
                if ids[i] == 1:
                    index = i
                    break

            position = (marker_locations[index][0], marker_locations[index][1])
            rotation = marker_angles[index]

            if bot1.next == 'NONE':
                bot1.position = position
                bot1.rotation = rotation
                bot1 = final_path(bot1, img, 0, 8, 8, 1)
                bot1 = target(bot1)
                bot1.target_position = get_position(bot1.target)

            if within(position, get_position(bot1.target)) and bot1.next == 'END':
                bot1.complete = True
                break

            elif within(position, get_position(bot1.target)):
                time.sleep(1)
                if bot1.next == 'RIGHT':
                    right(bot1)
                elif bot1.next == 'LEFT':
                    left(bot1)
                elif bot1.next == 'BACK':
                    back(bot1)
                elif bot1.next == 'PAY':
                    pay(bot1)
                print(bot1.next)
                print('-----')
                bot1.orientation = bot1.next_orientation
                bot1.position = position
                bot1.rotation = rotation
                bot1 = target(bot1)
                bot1.target_position = get_position(bot1.target)
                continue

            bot1 = pid(bot1, position)
            bot1 = pid_r(bot1, rotation)
            print ("Angle: {0:.2f} degrees".format(rotation))
            print ("Position: ",position)
            current = time.time()
            seconds = current - start
            print ("Time: {0:.2f} seconds".format(seconds))
            print ("Rotation Error: {0:.2f} degrees".format(bot1.error_r))
            print ("Position Error: {0:.2f} units".format(bot1.error))
            print ("Target: ", bot1.target_position)
            print("Orientation: ", bot1.orientation)
            print("Next: ", bot1.next)

            driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
            driver.get("http://192.168.11.6/forward?r="+str(bot1.right_enable)+"&l="+str(bot1.left_enable))
            driver.close()

            print("r="+str(bot1.right_enable)+", l="+str(bot1.left_enable))
            print('-----')

            bot1.right_enable = 220
            bot1.left_enable = 200

            cv2.imwrite('/home/vishwesh/Desktop/Geekbot/view2.jpg',img)
            draw_label(img, "{0:.2f}".format(seconds), (70,70), (0,0,0))
            out.write(img)


    os.system("nmcli con up GeekNetwork2")

    while (bot2.complete == False):
        for i in range(7):
            ret, frame = cap.read()
            if i == 6:
                break

        if ret == True:
            alpha = 2 # Contrast control (1.0-3.0)
            beta = 10 # Brightness control (0-100)

            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
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
                marker_locations.append(np.mean(marker, axis=0).astype('int'))
                marker_angles.append(angle)

            for pos in marker_locations:
                cv2.circle(img, (pos[0], pos[1]), 8, (0, 255, 0), -1)

            try:
                ids = ids.flatten()

            except AttributeError:
                print('ArUco marker was not detected')
                continue

            for i in range(len(ids)):
                if ids[i] == 2:
                    index = i
                    break

            position = (marker_locations[index][0], marker_locations[index][1])
            rotation = marker_angles[index]

            if bot2.next == 'NONE':
                bot2.position = position
                bot2.rotation = rotation
                bot2.route_path = [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (9, 10), (8, 10), (7, 10), (6, 10), (5, 10), (4, 10), (3, 10), (2, 10)]
                bot2 = draw_path(img, bot2, 2, 10, 10, 1)
                bot2.route_path = [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (9, 10), (8, 10), (7, 10), (6, 10), (5, 10), (4, 10), (3, 10), (2, 10), (0, 0), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1)]
                bot2 = target(bot2)
                bot2.target_position = get_position(bot2.target)

            if within(position, get_position(bot2.target)) and bot2.next == 'END':
                bot2.complete = True
                break

            elif within(position, get_position(bot2.target)):
                time.sleep(1)
                if bot2.next == 'RIGHT':
                    right(bot2)
                elif bot2.next == 'LEFT':
                    left(bot2)
                elif bot2.next == 'BACK':
                    back(bot2)
                elif bot2.next == 'PAY':
                    pay(bot2)
                print(bot2.next)
                print('-----')
                bot2.orientation = bot2.next_orientation
                bot2.position = position
                bot2.rotation = rotation
                bot2 = target(bot2)
                bot2.target_position = get_position(bot2.target)
                continue

            bot2 = pid(bot2, position)
            bot2 = pid_r(bot2, rotation)
            print ("Angle: {0:.2f} degrees".format(rotation))
            print ("Position: ",position)
            current = time.time()
            seconds = current - start
            print ("Time: {0:.2f} seconds".format(seconds))
            print ("Rotation Error: {0:.2f} degrees".format(bot2.error_r))
            print ("Position Error: {0:.2f} units".format(bot2.error))
            print ("Target: ", bot2.target_position)
            print("Orientation: ", bot2.orientation)
            print("Next: ", bot2.next)

            driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
            driver.get("http://192.168.11.6/forward?r="+str(bot2.left_enable)+"&l="+str(bot2.right_enable))
            driver.close()

            print("l="+str(bot2.right_enable)+", r="+str(bot2.left_enable))
            print('-----')

            bot2.right_enable = 200
            bot2.left_enable = 200

            cv2.imwrite('/home/vishwesh/Desktop/Geekbot/view2.jpg',img)
            draw_label(img, "{0:.2f}".format(seconds), (70,70), (0,0,0))
            out.write(img)

    os.system("nmcli con up GeekNetwork3")

    while (bot3.complete == False):
        for i in range(7):
            ret, frame = cap.read()
            if i == 6:
                break

        if ret == True:
            alpha = 2 # Contrast control (1.0-3.0)
            beta = 10 # Brightness control (0-100)

            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
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
                marker_locations.append(np.mean(marker, axis=0).astype('int'))
                marker_angles.append(angle)

            for pos in marker_locations:
                cv2.circle(img, (pos[0], pos[1]), 8, (0, 255, 0), -1)

            try:
                ids = ids.flatten()

            except AttributeError:
                print('ArUco marker was not detected')
                continue

            for i in range(len(ids)):
                if ids[i] == 3:
                    index = i
                    break

            position = (marker_locations[index][0], marker_locations[index][1])
            rotation = marker_angles[index]

            if bot3.next == 'NONE':
                bot3.position = position
                bot3.rotation = rotation
                bot3.route_path = [(11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (12, 10) , (13, 10), (14, 10), (15, 10), (16, 10), (17, 10), (18, 10), (19, 10)]
                bot3 = draw_path(img, bot3, 19, 10, 11, 1)
                bot3.route_path = [(11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (12, 10) , (13, 10), (14, 10), (15, 10), (16, 10), (17, 10), (18, 10), (19, 10), (0, 0), (19, 10), (18, 10), (17, 10), (16, 10), (15, 10), (14, 10), (13, 10), (12, 10), (11, 10), (11, 9), (11, 8), (11, 7), (11, 6), (11, 5), (11, 4), (11, 3), (11, 2), (11, 1)]
                bot3 = target(bot3)
                bot3.target_position = get_position(bot3.target)

            if within(position, get_position(bot3.target)) and bot3.next == 'END':
                bot3.complete = True
                break

            elif within(position, get_position(bot3.target)):
                time.sleep(1)
                if bot3.next == 'RIGHT':
                    right(bot3)
                elif bot3.next == 'LEFT':
                    left(bot3)
                elif bot3.next == 'BACK':
                    back(bot3)
                elif bot3.next == 'PAY':
                    pay(bot3)
                print(bot3.next)
                print('-----')
                bot3.orientation = bot3.next_orientation
                bot3.position = position
                bot3.rotation = rotation
                bot3 = target(bot3)
                bot3.target_position = get_position(bot3.target)
                continue

            bot3 = pid(bot3, position)
            bot3 = pid_r(bot3, rotation)
            print ("Angle: {0:.2f} degrees".format(rotation))
            print ("Position: ",position)
            current = time.time()
            seconds = current - start
            print ("Time: {0:.2f} seconds".format(seconds))
            print ("Rotation Error: {0:.2f} degrees".format(bot3.error_r))
            print ("Position Error: {0:.2f} units".format(bot3.error))
            print ("Target: ", bot3.target_position)
            print("Orientation: ", bot3.orientation)
            print("Next: ", bot3.next)

            driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
            driver.get("http://192.168.11.6/forward?r="+str(bot3.right_enable)+"&l="+str(bot3.left_enable))
            driver.close()

            print("r="+str(bot3.right_enable)+", l="+str(bot3.left_enable))
            print('-----')

            bot3.right_enable = 200
            bot3.left_enable = 200

            cv2.imwrite('/home/vishwesh/Desktop/Geekbot/view2.jpg',img)
            draw_label(img, "{0:.2f}".format(seconds), (70,70), (0,0,0))
            out.write(img)

    os.system("nmcli con up GeekNetwork4")

    while (bot4.complete == False):
        for i in range(7):
            ret, frame = cap.read()
            if i == 6:
                break

        if ret == True:
            alpha = 2 # Contrast control (1.0-3.0)
            beta = 10 # Brightness control (0-100)

            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
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
                marker_locations.append(np.mean(marker, axis=0).astype('int'))
                marker_angles.append(angle)

            for pos in marker_locations:
                cv2.circle(img, (pos[0], pos[1]), 8, (0, 255, 0), -1)

            try:
                ids = ids.flatten()

            except AttributeError:
                print('ArUco marker was not detected')
                continue

            for i in range(len(ids)):
                if ids[i] == 4:
                    index = i
                    break

            position = (marker_locations[index][0], marker_locations[index][1])
            rotation = marker_angles[index]

            if bot4.next == 'NONE':
                bot4.position = position
                bot4.rotation = rotation
                bot4 = final_path(bot4, img, 0, 8, 8, 1)
                bot4 = target(bot4)
                bot4.target_position = get_position(bot4.target)

            if within(position, get_position(bot4.target)) and bot4.next == 'END':
                bot4.complete = True
                break

            elif within(position, get_position(bot4.target)):
                time.sleep(1)
                if bot4.next == 'RIGHT':
                    right(bot4)
                elif bot4.next == 'LEFT':
                    left(bot4)
                elif bot4.next == 'BACK':
                    back(bot4)
                elif bot4.next == 'PAY':
                    pay(bot4)
                print(bot4.next)
                print('-----')
                bot4.orientation = bot4.next_orientation
                bot4.position = position
                bot4.rotation = rotation
                bot4 = target(bot4)
                bot4.target_position = get_position(bot4.target)
                continue

            bot4 = pid(bot4, position)
            bot4 = pid_r(bot4, rotation)
            print ("Angle: {0:.2f} degrees".format(rotation))
            print ("Position: ",position)
            current = time.time()
            seconds = current - start
            print ("Time: {0:.2f} seconds".format(seconds))
            print ("Rotation Error: {0:.2f} degrees".format(bot4.error_r))
            print ("Position Error: {0:.2f} units".format(bot4.error))
            print ("Target: ", bot4.target_position)
            print("Orientation: ", bot4.orientation)
            print("Next: ", bot4.next)

            driver = webdriver.Chrome(executable_path='/home/vishwesh/Desktop/Geekbot/chromedriver')
            driver.get("http://192.168.11.6/forward?r="+str(bot4.right_enable)+"&l="+str(bot4.left_enable))
            driver.close()

            print("r="+str(bot4.right_enable)+", l="+str(bot4.left_enable))
            print('-----')

            bot4.right_enable = 200
            bot4.left_enable = 200

            cv2.imwrite('/home/vishwesh/Desktop/Geekbot/view2.jpg',img)
            draw_label(img, "{0:.2f}".format(seconds), (70,70), (0,0,0))
            out.write(img)

    cv2.waitKey(0)
    cap.release()
    out.release()
    cv2.destroyAllWindows()