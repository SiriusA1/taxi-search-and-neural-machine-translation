# Conor McGullam
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from operator import itemgetter
from geopy.distance import geodesic

def CalculateDistance(data, s_long, s_lat, t_long, t_lat):
    ret1 = data['trip_distance'][(data['pickup_longitude']==s_long) & (data['pickup_latitude']==s_lat) & (data['dropoff_longitude']==t_long) & (data['dropoff_latitude']==t_lat)]
    ret2 = data['trip_distance'][(data['pickup_longitude']==t_long) & (data['pickup_latitude']==t_lat) & (data['dropoff_longitude']==s_long) & (data['dropoff_latitude']==s_lat)]
    if len(ret1) > 0:
        return ret1.iloc[0]
    else:
        return ret2.iloc[0]

def FindNeighbors(edges, nodeid):
    neighbors = []

    for i in range(len(edges)):
        if int(edges["nodeid1"][i]) == int(nodeid):
            neighbors.append(edges["nodeid2"][i])
        elif int(edges["nodeid2"][i]) == int(nodeid):
            neighbors.append(edges["nodeid1"][i])
    return neighbors

def UniformCostSearch(data, nodes, edges, start, target):
    print("------------Uniform Cost Search------------")
    start_time = time.time()

    opened = []
    closed = []
    dist = 0
    steps = 0

    opened.append((start, 0))
    while True:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Steps ", steps)
        if len(opened) == 0:
            print(f"No solution found. {steps} steps taken.")
            break
    
        opened = sorted(opened,key=itemgetter(1))
        curr_node = opened.pop(0)
        print("current ", curr_node)
        closed.append(curr_node)
        
        # check if the node id is the solution id
        if int(curr_node[0]) == int(target):
            dist = curr_node[1]
            break

        # fetch all connected nodes (neighbors is a list of node ids)
        neighbors = FindNeighbors(edges, curr_node[0])
        print("neighbors ", neighbors)
        # calculate distances of neighbors and add them to opened
        if len(neighbors) > 0:
            for neighbor in neighbors:
                temp_open = [ (id, dist) for id, dist in opened if int(id)  == int(neighbor) ]
                temp_close = [ (id, dist) for id, dist in closed if int(id)  == int(neighbor) ]
                n_dist = curr_node[1] + CalculateDistance(data, nodes["long"][nodes["nodeid"]==int(curr_node[0])].iloc[0], nodes["lat"][nodes["nodeid"]==int(curr_node[0])].iloc[0], nodes["long"][nodes["nodeid"]==neighbor].iloc[0], nodes["lat"][nodes["nodeid"]==neighbor].iloc[0])
                if len(temp_close) == 0 and len(temp_open) == 0:
                    opened.append((neighbor, n_dist))
                elif len(temp_open) > 0:
                    old_node = temp_open[0]
                    if n_dist < old_node[1]:
                        opened.remove(old_node)
                        opened.append((neighbor, n_dist))
        steps += 1
    return dist, (time.time() - start_time), closed

def AstarSearch(data, nodes, edges, start, target):
    print("------------A* Search------------")
    start_time = time.time()
    coords1 = (nodes["lat"][nodes["nodeid"]==int(start)].iloc[0], nodes["long"][nodes["nodeid"]==int(start)].iloc[0])
    coords2 = (nodes["lat"][nodes["nodeid"]==int(target)].iloc[0], nodes["long"][nodes["nodeid"]==int(target)].iloc[0])
    heuristic = geodesic(coords1, coords2).mi
    opened = []
    closed = []
    dist = 0
    steps = 0

    opened.append((start, heuristic))
    while True:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Steps ", steps)
        if len(opened) == 0:
            print(f"No solution found. {steps} steps taken.")
            break
    
        opened = sorted(opened,key=itemgetter(1))
        curr_node = opened.pop(0)
        
        print("current ", curr_node)
        closed.append(curr_node)
        
        # check if the node id is the solution id
        if int(curr_node[0]) == int(target):
            dist = curr_node[1]
            break

        # fetch all connected nodes (neighbors is a list of node ids)
        neighbors = FindNeighbors(edges, curr_node[0])
        print("neighbors ", neighbors)
        # calculate distances of neighbors and add them to opened
        if len(neighbors) > 0:
            for neighbor in neighbors:
                temp_open = [ (id, dist) for id, dist in opened if int(id)  == int(neighbor) ]
                temp_close = [ (id, dist) for id, dist in closed if int(id)  == int(neighbor) ]
                n_dist = curr_node[1] + CalculateDistance(data, nodes["long"][nodes["nodeid"]==int(curr_node[0])].iloc[0], nodes["lat"][nodes["nodeid"]==int(curr_node[0])].iloc[0], nodes["long"][nodes["nodeid"]==neighbor].iloc[0], nodes["lat"][nodes["nodeid"]==neighbor].iloc[0])
                coords1 = (nodes["lat"][nodes["nodeid"]==neighbor].iloc[0], nodes["long"][nodes["nodeid"]==neighbor].iloc[0])
                heuristic = geodesic(coords1, coords2).mi
                if len(temp_close) == 0 and len(temp_open) == 0:
                    opened.append((neighbor, n_dist + heuristic))
                elif len(temp_open) > 0:
                    old_node = temp_open[0]
                    if (n_dist + heuristic) < old_node[1]:
                        opened.remove(old_node)
                        opened.append((neighbor, n_dist + heuristic))
        steps += 1
    return dist, (time.time() - start_time), closed

def MakeGraphs(data):
    nodes = []
    ids = []
    edges = []
    nodeid = 0
    for i in range(len(data)):
        # variables below used for adding edges
        # if pickup already exists in list:
        # # # need to find existing pickup by getting index of the (long, lat) 
        # # # this index is equal to the node's id
        # # # if dropoff does not exist, add dropoff to nodes and add new edge with new id and previously fetched pickup id
        # if pickup does not exist: 
        # # # either connect to new dropoff with 2 new ids or find existing dropoff (same as above) and add new edge
        # if both exist, edge will already exist and nothing needs to be done
        pickup_exists = False
        pickup_id = 0
        dropoff_exists = False
        dropoff_id = 0
        if (data["pickup_longitude"][i], data["pickup_latitude"][i]) not in nodes:
            nodes.append((data["pickup_longitude"][i], data["pickup_latitude"][i]))
            ids.append(nodeid)
            pickup_id = nodeid
            nodeid+=1
        else:
            pickup_id = nodes.index((data["pickup_longitude"][i], data["pickup_latitude"][i]))
            pickup_exists = True
            
        if (data["dropoff_longitude"][i], data["dropoff_latitude"][i]) not in nodes:
            nodes.append((data["dropoff_longitude"][i], data["dropoff_latitude"][i]))
            ids.append(nodeid)
            dropoff_id = nodeid
            nodeid+=1
        else:
            dropoff_id = nodes.index((data["dropoff_longitude"][i], data["dropoff_latitude"][i]))
            dropoff_exists = True

        if not (pickup_exists and dropoff_exists):
            edges.append((pickup_id, dropoff_id))

    nodes_df = pd.DataFrame(nodes, columns = ['long', 'lat'])
    nodes_df.insert(0, 'nodeid', ids)
    edges_df = pd.DataFrame(edges, columns = ['nodeid1', 'nodeid2'])

    return nodes_df, edges_df
    

def main():
    data = pd.read_csv('nyc_taxi_data.csv')
    #nodes, edges = MakeGraphs(data)
    #nodes.to_csv("nodes.csv")
    #edges.to_csv("edges.csv")
   
    nodes = pd.read_csv('nodes.csv')
    edges = pd.read_csv('edges.csv')
    
    nodeid1 = input("Enter node id 1: ")
    nodeid2 = input("Enter node id 2: ")

    distance1, time1, path1 = UniformCostSearch(data, nodes, edges, int(nodeid1), int(nodeid2))

    distance2, time2, path2 = AstarSearch(data, nodes, edges, int(nodeid1), int(nodeid2))

    trimmed_path1 = []
    for (id, dist) in path1:
        trimmed_path1.append(id)
    trimmed_path2 = []
    for (id, dist) in path2:
        trimmed_path2.append(id)

    print("Path found with distance ", distance1, "in time ", time1)
    print("Path: ", trimmed_path1)

    print("Path found with distance ", distance2, "in time ", time2)
    print("Path: ", trimmed_path2)

if __name__ == "__main__":
    main()