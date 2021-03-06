import numpy as np 
import argparse 
import pandas as pd 
import xml.etree.ElementTree as et 
import random 

'''
Makes .rou.xml file for given .net.xml file and simulation length.

Std Inputs:
--------
-n location of .net.xml file
-r name of route file
-t simulation length

'''

prs = argparse.ArgumentParser(formatter_class=
                                       argparse.ArgumentDefaultsHelpFormatter)
prs.add_argument("-n", dest="network", required=True, 
                               help="The .net.xml file to make routes for.\n")
prs.add_argument("-r", dest="route", required=True,
                              help="The name of the .rou.xml file to make.\n")
prs.add_argument("-t", dest="time", required=False, default=2000000, type=int, 
                                       help="The length of the simulation.\n")
args = prs.parse_args()

if args.time < 1000:
    raise "Error: Time must be greater than 1000."
if args.route[-8:] != ".rou.xml":
    raise "Error: Route file must end in .rou.xml!"
if args.network[-8:] != ".net.xml":
    raise "Error: Network file must end in .net.xml!"

tree = et.parse(str(args.network))
root = tree.getroot()
tl_ids = []
edges = []; edges_to = []; edges_from = []

for elem in root:
    # find tl_ids from net file
    if elem.tag == "junction":
        if elem.attrib.get("type")== "traffic_light":
            tl_ids = np.append(tl_ids, int(elem.attrib.get("id")))

    # find all network edges 
    if elem.tag == "edge":
        if elem.attrib.get("function") != "internal":
            edges = np.append(edges, elem.attrib.get("id"))
            edges_from = np.append(edges_from, int(elem.attrib.get("from")))
            edges_to = np.append(edges_to, int(elem.attrib.get("to")))

ext_edges = []; ext_to = [] 
ent_edges = []; ent_from = []
# find possible entrance / exits 
for edge in range(0,len(edges)):
    for tl in tl_ids:
        if edges_to[edge] == tl:
            break
        if tl == tl_ids[-1]:
            ext_edges = np.append(ext_edges, edges[edge])
            ext_to = np.append(ext_to, edges_to[edge])

    for tl in tl_ids:
        if edges_from[edge] == tl:
            break
        if tl == tl_ids[-1]:
            ent_edges = np.append(ent_edges, edges[edge])
            ent_from = np.append(ent_from, edges_from[edge])


# possible entrance/exit edge combos
origins = []; destinations = []
for ent in range(0,len(ent_edges)):
    for exi in range(0,len(ext_edges)):
        if ent_from[ent] != ext_to[exi]:
            origins = np.append(origins,ent_edges[ent])
            destinations = np.append(destinations,ext_edges[exi])
    

f = open(str(args.route), 'w')

f.write("<routes>\n")
for routes in range(0,len(origins)):
    f.write("<trip id='trip_{}_{}' depart='0' "\
            "from='{}' to='{}' />\n".format(origins[routes],
                                            destinations[routes],
                                            origins[routes],
                                            destinations[routes]))

for time in range(0,args.time,1000):
    for routes in range(0,len(origins)):
        var = random.randint(0,5)
        sign = random.randint(-1,1)
        f.write("<flow id='flow_{}_{}_{}' ".format(origins[routes],
                                                   destinations[routes],
                                                   time))
        f.write("begin='{}' end='{}' number='{}'".format(time, 
                                                         time+1000, 
                                                         8 + sign*var))
        f.write("from='{}' to='{}'".format(origins[routes],
                                           destinations[routes]))
        f.write("departSpeed='max' departPos='base' departLane='best'/> \n")

f.write("</routes>\n")
f.close()
