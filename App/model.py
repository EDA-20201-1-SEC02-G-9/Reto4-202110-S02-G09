"""
 * Copyright 2020, Departamento de sistemas y Computación,
 * Universidad de Los Andes
 *
 *
 * Desarrolado para el curso ISIS1225 - Estructuras de Datos y Algoritmos
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along withthis program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contribuciones:
 *
 * Dario Correal - Version inicial
 """


import config as cf
from DISClib.ADT import list as lt
from DISClib.ADT import map as mp
from DISClib.ADT import graph as gp
from DISClib.DataStructures import mapentry as me
from DISClib.DataStructures import arraylistiterator as al_it
from DISClib.DataStructures import linkedlistiterator as ll_it
from DISClib.Algorithms.Graphs import dijsktra as dij
from DISClib.Algorithms.Graphs import scc
from DISClib.Algorithms.Graphs import prim
from math import pi, sin, cos, asin
assert cf

"""
Se define la estructura de un catálogo de videos. El catálogo tendrá dos listas, una para los videos, otra para las categorias de
los mismos.
"""
# Clasificación de datos

def special_split(line:str):
    divided = []
    quote = False
    temp = ''
    for c in line:
        if c == ',' and not quote:
            if ' km' in temp:
                temp = temp.replace(",","").replace(" km","")
            divided.append(temp)
            temp = ''
        elif c=='"':
            quote = not quote
        else:
            temp += c
    if temp:
        divided.append(temp)
    return divided

def data_type(text:str):
    if text.isnumeric():
        return 'i'
    elif text.replace(".","").replace("-","").isnumeric():
        return 'f'
    else:
        return 's'

def correct_type(text:str, t: str):
    if t == 'i':
        if text == 'n.a.':
            text = '0'
        return int(text)
    elif t == 'f':
        if text == 'n.a.':
            text = '0'
        return float(text)
    else:
        return text

def line_types(line: str):
    params = special_split(line.replace("\n",""))
    types = lt.newList(datastructure='ARRAY_LIST')
    for param in params:
        t = data_type(param)
        lt.addLast(types, t)
    return types


class landing_points:

    def add_point(self, line:str, types: list):
        params = special_split(line.replace("\n",""))
        characteristics = lt.newList(datastructure='ARRAY_LIST')
        i = 1
        for param in params:
            t = lt.getElement(types, i)
            char = correct_type(param, t)
            lt.addLast(characteristics, char)
            i += 1
        id = lt.getElement(characteristics, 1)
        name = lt.getElement(characteristics, 3).split(",")[0]
        mp.put(self.points_by_id, id, characteristics)
        mp.put(self.points_by_name, name, characteristics)
    
    def add_country(self, line: str, types: list):
        params = special_split(line.replace("\n",""))
        name = params[0]
        characteristics = lt.newList(datastructure='ARRAY_LIST')
        i = 2
        for param in params[1:]:
            t = lt.getElement(types, i)
            char = correct_type(param, t)
            lt.addLast(characteristics, char)
            i += 1
        mp.put(self.countries, name, characteristics)

    def add_connection(self, line:str, types: list):
        params = special_split(line.replace("\n",""))
        characteristics = lt.newList(datastructure='ARRAY_LIST')
        i = 1
        for param in params:
            t = lt.getElement(types, i)
            char = correct_type(param, t)
            lt.addLast(characteristics, char)
            i += 1
        lt.addLast(self.connections_list, characteristics)

    # Funciones para creacion de datos
    
    def open_points(self, filepath: str):
        self.points_by_id = mp.newMap(numelements=1279)
        self.points_by_name = mp.newMap(numelements=1279)
        file = open(filepath, 'r')
        file.readline()
        line = file.readline()
        types = line_types(line)
        while line:
            self.add_point(line, types)
            line = file.readline()
        file.close()

    def open_connections(self, filepath: str):
        self.connections_list = lt.newList(datastructure='ARRAY_LIST')
        file = open(filepath, 'r')
        file.readline()
        line = file.readline()
        types = line_types(line)
        lt.changeInfo(types, 5, 'f')
        lt.changeInfo(types, 6, 's')
        lt.changeInfo(types, 8, 'f')
        while line:
            self.add_connection(line, types)
            line = file.readline()
        file.close()

    def open_countries(self, filepath:str):
        self.countries = mp.newMap(numelements=236)
        file = open(filepath, 'r')
        file.readline()
        line = file.readline()
        types = line_types(line)
        while line:
            self.add_country(line, types)
            line = file.readline()
        file.close()

    
    # Optimizaciones de requerimientos
    def vertex_cluster(self, vertex):
        return mp.get(self.clusters['idscc'], vertex)['value']

    def cluster_mas_grande(self):
        tamaños = lt.newList(datastructure="ARRAY_LIST")
        ref = lt.newList(datastructure='ARRAY_LIST')
        for i in range(self.cluster_number):
            lt.addLast(tamaños, 0)
            lt.addLast(ref, 0)
        vertices = gp.vertices(self.connections_map)
        vertex_it = ll_it.newIterator(vertices)
        while ll_it.hasNext(vertex_it):
            vertex = ll_it.next(vertex_it)
            cluster = self.vertex_cluster(vertex)
            new_value = lt.getElement(tamaños, cluster) + 1
            lt.changeInfo(tamaños, cluster, new_value)
            if not lt.getElement(ref, cluster):
                lt.changeInfo(ref, cluster, vertex)
        maximo = 0, 0
        for i in range(1, self.cluster_number+1):
            if maximo[1] < lt.getElement(tamaños, i):
                maximo = i, lt.getElement(tamaños, i)
        self.mas_grande = *maximo, lt.getElement(ref, maximo[0])


    def req_1_optimization(self):
        self.clusters = scc.KosarajuSCC(self.connections_map)
        self.cluster_number = scc.connectedComponents(self.clusters)
        
    
    # Construccion de modelos
    def assign_points_to_countries(self):
        self.point_country = mp.newMap(numelements=236)
        point_it = ll_it.newIterator(mp.valueSet(self.points_by_id))
        while ll_it.hasNext(point_it):
            point = ll_it.next(point_it)
            city = lt.getElement(point, 3)
            country = city.split(", ") [-1]
            id = lt.getElement(point, 1)
            pair = mp.get(self.point_country, country)
            if pair:
                lt.addLast(pair['value'], id)
            else:
                new_list = lt.newList()
                lt.addLast(new_list, id)
                mp.put(self.point_country, country, new_list)

    def haversine_2(self, id_1, id_2):
        land_1 = mp.get(self.points_by_id, id_1)['value']
        land_2 = mp.get(self.points_by_id, id_2)['value']
        lat_1 = lt.getElement(land_1, 4) * pi / 180
        lat_2 = lt.getElement(land_2, 4) * pi / 180
        lon_1 = lt.getElement(land_1, 5) * pi / 180
        lon_2 = lt.getElement(land_2, 5) * pi / 180
        respuesta = 12742 * asin(((sin((lat_2 - lat_1) / 2)) ** 2 + cos(lat_2) * cos(lat_1) * (sin((lon_2 - lon_1) / 2)) ** 2) ** 0.5)
        return respuesta
    
    def haversine_1(self, lat_1,lon_1, id_2):
        lat_1 *= pi / 180
        lon_1 *= pi/ 180
        land_2 = mp.get(self.points_by_id, id_2)['value']
        lat_2 = lt.getElement(land_2, 4) * pi / 180
        lon_2 = lt.getElement(land_2, 5) * pi / 180
        respuesta = 12742 * asin(((sin((lat_2 - lat_1) / 2)) ** 2 + cos(lat_2) * cos(lat_1) * (sin((lon_2 - lon_1) / 2)) ** 2) ** 0.5)
        return respuesta

    def draw_connections(self):
        c_map = gp.newGraph(size=lt.size(self.connections_list), directed=True)
        vertex_it = ll_it.newIterator(mp.valueSet(self.points_by_id))
        edge_it = al_it.newIterator(self.connections_list)
        while ll_it.hasNext(vertex_it):
            new_vertex = lt.getElement(ll_it.next(vertex_it), 1)
            gp.insertVertex(c_map, new_vertex)
        while al_it.hasNext(edge_it):
            temp = al_it.next(edge_it)
            new_edge = lt.getElement(temp, 1), lt.getElement(temp, 2)
            gp.addEdge(c_map, *new_edge, self.haversine_2(*new_edge))
        self.connections_map = c_map
    
    def __init__(self, filepath_points: str, filepath_connections: str, filepath_countries: str):
        (self.points_by_id, self.points_by_name, self.connections_list, self.countries,
        self.connections_map, self.point_country, self.clusters, self.cluster_number,
        self.mas_grande) = None, None, None, None, None, None, None, None, None
        self.open_points(filepath_points)
        self.open_connections(filepath_connections)
        self.open_countries(filepath_countries)
        self.assign_points_to_countries()
        self.draw_connections()
        self.req_1_optimization()
        self.cluster_mas_grande()
        print(self.mas_grande)
    
    # Requerimientos
    def req_1(self, lp1, lp2):
        land_1 = mp.get(self.points_by_name, lp1)['value']
        id_1 = lt.getElement(land_1, 1)
        land_2 = mp.get(self.points_by_name, lp2)['value']
        id_2 = lt.getElement(land_2, 1)
        return scc.stronglyConnected(self.clusters, id_1, id_2)

    def req_2(self):
        respuesta = []
        vertices = gp.vertices(self.connections_map)
        vertex_it = ll_it.newIterator(vertices)
        while ll_it.hasNext(vertex_it):
            vertex = ll_it.next(vertex_it)
            point = mp.get(self.points_by_id,vertex)['value']
            name = lt.getElement(point, 3)
            edges = gp.degree(self.connections_map, vertex)
            respuesta.append("{} ({}): {} conexiones".format(name,vertex,edges))
        return "\n".join(respuesta)

    def nearest_point_to_capital(self, country):
        points = mp.get(self.point_country, country)['value']
        point_it = ll_it.newIterator(points)
        capital = mp.get(self.countries, country)['value']
        lat_cap = lt.getElement(capital, 2)
        lon_cap = lt.getElement(capital, 3)
        minimum = 0, 600000
        while ll_it.hasNext(point_it):
            point = ll_it.next(point_it)
            dist = self.haversine_1(lat_cap, lon_cap, point)
            if dist < minimum[1]:
                minimum = point, dist
        return minimum

    def path_str(self, path):
        respuesta = []
        path_it = ll_it.newIterator(path)
        while ll_it.hasNext(path_it):
            edge = ll_it.next(path_it)
            pointA = mp.get(self.points_by_id, edge['vertexA'])['value']
            pointB = mp.get(self.points_by_id, edge['vertexB'])['value']
            dist = round(edge['weight'], 2)
            cityA = lt.getElement(pointA, 3).split(",")[0]
            cityB = lt.getElement(pointB, 3).split(",")[0]
            respuesta.append("{} - {}: {} km".format(cityA,cityB,dist))
        return "\n".join(respuesta)
    
    def req_3(self, countryA, countryB):
        pointA, distA = self.nearest_point_to_capital(countryA)
        pointB, distB = self.nearest_point_to_capital(countryB)
        capitalA = lt.getElement(mp.get(self.countries, countryA)['value'], 1)
        capitalB = lt.getElement(mp.get(self.countries, countryB)['value'], 1)
        search_map = dij.Dijkstra(self.connections_map, pointA)
        distC = dij.distTo(search_map, pointB)
        path = dij.pathTo(search_map, pointB)
        path = self.path_str(path)
        cityA = path.split(" -", 1)[0]
        cityB = path.split("- ")[-1].split(":")[0]
        path = (
            "{} - {}: {} km\n".format(capitalA, cityA, round(distA,2)) +
            path +
            "\n{} - {}: {} km".format(cityB,capitalB,round(distB,2)))
        dist = distA + distB + distC
        return path, round(dist,2)

    def req_4(self):
        search = prim.PrimMST(self.connections_map)
        scan = prim.scan(self.connections_map, search, self.mas_grande[2])
        

    def req_5(self, lp):
        pass

    def req_6(self, country, cable):
        pass

    def req_7(self, IP1, IP2):
        pass

# Funciones para agregar informacion al catalogo

# Funciones de consulta

# Funciones utilizadas para comparar elementos dentro de una lista

# Funciones de ordenamiento

if __name__ == '__main__':
    ld = landing_points("Data/landing_points.csv", "Data/connections.csv","Data/countries.csv")
    print(ld.req_4())
    print(ld.vertex_cluster(14844))