from gym import spaces
import numpy as np
# a1 = spaces.Discrete(8)
# a2 = spaces.Discrete(6)
# action_space= [a1,a2]

# dictiona = {'state': [[0 for j in range(action_space[1].n)] for i in range(action_space[0].n)]} 

# print(dictiona)

# a= [[10, 15, 10],[13, 14, 15]]
# print( np.argmax(a, axis=0))
# print( np.argmax(a, axis=1))


# coord_graph = {
#     1:[2,6],
#     2:[1,5],
#     5:[2,6],
#     6:[1,5]
# }
# # remove any duplicates from coord graph
# # coord edges represent the edges as a vector where 
# coord_edges = []
# vertex_list = list(coord_graph.keys())
# print(vertex_list)
# for vertex in coord_graph:
#     for i in range(0,len(coord_graph[vertex])):
#         if coord_graph[vertex][i] in vertex_list:
#             coord_edges = np.append(coord_edges, vertex)
#             coord_edges = np.append(coord_edges, coord_graph[vertex][i])
#     vertex_list.remove(vertex)


# print('coord edges')
# print(coord_edges)
agent = 2
variables = np.array([1,2,5])
variables = variables[variables != agent]

print(variables)