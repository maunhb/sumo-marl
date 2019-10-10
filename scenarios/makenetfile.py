gridsize = 4

junctions = [x for x in range(1+gridsize^2)]
traffic_lights = []

inner_edges = []
outer_edges =[]

f = open('{}x{}.net.xml'.format(gridsize), 'w')
# preamble
f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
f.write('<net version="1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">\n')
f.write('<location netOffset="250.00,550.00" convBoundary="0.00,0.00,800.00,800.00" origBoundary="-250.00,-550.00,550.00,250.00" projParameter="!"/>\n')
# write edges

# write more edges 
for junction_id in junctions:
    for edge in inner_edges:
        f.write("<edge id='{}{}{}' from='{} 'to='{}' priority='3'>\n".format(junction_id, direction, edgetype, from_junction, to_junction))
        f.write("lane id='{}{}{}_0' index='0' speed='13.89' length='129.20' shape='{}'/>".format(junction_id, direction, edgetype, shape))
        f.write("lane id='{}{}{}_1' index='1' speed='13.89' length='129.20' shape='{}'/>".format(junction_id, direction, edgetype, shape))
        f.write('</edge>\n')
    for edge in outer_edges:
        f.write("<edge id='{}{}{}' from='{} 'to='{}' priority='3'>\n".format(junction_id, direction, edgetype, from_junction, to_junction))
        f.write("lane id='{}{}{}_0' index='0' speed='13.89' length='239.60' shape='{}'/>".format(junction_id, direction, edgetype, shape))
        f.write("lane id='{}{}{}_1' index='1' speed='13.89' length='239.60' shape='{}'/>".format(junction_id, direction, edgetype, shape))
        f.write('</edge>\n')

for junction_id in traffic_lights:
    f.write('<tlLogic id="{}" type="static" programID="0" offset="0">\n'.format(junction_id))
    f.write('<phase duration="32" state="GGrrrrGGrrrr"/>\n')
    f.write('<phase duration="3"  state="yyrrrryyrrrr"/>\n')
    f.write('<phase duration="32" state="rrGrrrrrGrrr"/>\n')
    f.write('<phase duration="3"  state="rryrrrrryrrr"/>\n')
    f.write('<phase duration="32" state="rrrGGrrrrGGr"/>\n')
    f.write('<phase duration="3"  state="rrryyrrrryyr"/>\n')
    f.write('<phase duration="32" state="rrrrrGrrrrrG"/>\n')
    f.write('<phase duration="3"  state="rrrrryrrrrry"/>\n')
    f.write('</tlLogic>\n')

for junction_id in traffic_lights:
    f.write('<junction id="{}" type="traffic_light" x="{}" y="{}" incLanes="{}" intLanes="{}" shape="{}">\n'.format(junction_id))
    f.write('<request index="0"  response="000000000000" foes="000000010000" cont="0"/>\n')
    f.write('<request index="1"  response="100000000000" foes="111100010000" cont="0"/>\n')
    f.write('<request index="2"  response="100010100000" foes="100010110000" cont="1"/>\n')
    f.write('<request index="3"  response="000010000000" foes="000010000000" cont="0"/>\n')
    f.write('<request index="4"  response="000010000111" foes="100010000111" cont="0"/>\n')
    f.write('<request index="5"  response="010110000100" foes="010110000100" cont="1"/>\n')
    f.write('<request index="6"  response="000000000000" foes="010000000000" cont="0"/>\n')
    f.write('<request index="7"  response="000000100000" foes="010000111100" cont="0"/>\n')
    f.write('<request index="8"  response="100000100010" foes="110000100010" cont="1"/>\n')
    f.write('<request index="9"  response="000000000010" foes="000000000010" cont="0"/>\n')
    f.write('<request index="10" response="000111000010" foes="000111100010" cont="0"/>\n')
    f.write('<request index="11" response="000100010110" foes="000100010110" cont="1"/>\n')
    f.write('</junction>\n')

for junction_id in exits:
    f.write('<junction id="{}" type="unregulated" x="{}" y="{}" incLanes="{}" intLanes="" shape="{}"/>\n'.format(junction_id))

for junction in unknown:
    f.write('<junction id="{}" type="internal" x="{}" y="{}" incLanes="{}" intLanes="{}"/>\n')

for connection in connections:
    f.write('<connection from="{}" to="{}" fromLane="{}" toLane="{}" via="{}" tl="{}" linkIndex="{}" dir="{}" state="o"/>\n')

for connection in otherconnection:
    f.write('<connection from="{}" to="{}" fromLane="{}" toLane="{}" dir="{}" state="M"/>\n')

for connection in viaconnection:
    f.write('<connection from="{}" to="{}" fromLane="{}" toLane="{}" via="{}" dir="{}" state="m"/>\n')

f.write("</net>\n")
f.close()