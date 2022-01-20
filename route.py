from cmath import sqrt
import csv

wall_map = []

amd = [(6,3), (6,4), (7,2), (7,5), (8,2), (8,5), (9,3), (9,4)]            #the eight coordinates for the drop zones of the cities
pune = [(10,3), (10,4), (11,2), (11,5), (12,2), (12,5), (13,3), (13,4)]
bng = [(6,7), (6,8), (7,6), (7,9), (8,6), (8,9), (9,7), (9,8)]
klt = [(2,11), (2,12), (3,10), (3,13), (4,10), (4,13), (5,11), (5,12)]
mum = [(10,11), (10,12), (11,10), (11,13), (12,10), (12,13), (13,11), (13,12)]
delh = [(7,11), (7,12), (8,10), (8,13), (9,10), (9,13), (10,11), (10,12)]
chen = [(10,7), (10,8), (11,6), (11,9), (12,6), (12,9), (13,7), (13,8)]
jai = [(2,3), (2,4), (3,2), (3,5), (4,2), (4,5), (5,3), (5,4)]
hyd = [(2,7), (2,8), (3,6), (3,9), (4,6), (4,9), (5,7), (5,8)]

file = open('Sample Data - Sheet1.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows1 = []          #destination cities for bot 1 and 2 respectively
rows2 = []
for row in csvreader:
    if(row[1] == '1'):
        rows1.append(row)
    else:
        rows2.append(row)

count1 = 0                          #records which package the bot has to deliver
count2 = 0

def dest(inp, sx, sy):              #the inp will be the row1[2] for 
    global count1
    global count2
    if inp == 'Ahmedabad':
        minDist = 100
        dest = None
        for coor in amd:
            if coor not in wall_map:
                dist = sqrt((sx - coor[0])**2 + (sy - coor[1])**2)
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest

    elif inp == 'Pune':
        minDist = 100
        dest = None
        for coor in pune:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest

    elif inp == 'Bengaluru':
        minDist = 100
        dest = None
        for coor in bng:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest
    
    elif inp == 'Kolkata':
        minDist = 100
        dest = None
        for coor in klt:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest

    elif inp == 'Mumbai':
        minDist = 100
        dest = None
        for coor in mum:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest
    
    elif inp == 'Delhi':
        minDist = 100
        dest = None
        for coor in delh:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest

    elif inp == 'Chennai':
        minDist = 100
        dest = None
        for coor in chen:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest
    
    elif inp == 'Jaipur':
        minDist = 100
        dest = None
        for coor in jai:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest
    
    elif inp == 'Hydrabad':
        minDist = 100
        dest = None
        for coor in hyd:
            if coor not in wall_map:
                dist = sqrt(sqrt((sx - coor[0])**2 + (sy - coor[1])**2))
                if dist < minDist:
                    minDist = dist
                    dest = coor
        if sx == 15 and sy == 10:
            count1 += 1
        else:
            count2 += 1
        return dest