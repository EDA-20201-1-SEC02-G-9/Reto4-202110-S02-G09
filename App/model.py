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
from DISClib.Algorithms.Sorting import mergesort
from math import pi, sin, cos, asin
from App.get_ip import get_location
import ipapi.ipapi as ipapi
import folium
assert cf

"""
Se define la estructura de un catálogo de videos. El catálogo tendrá dos listas, una para los videos, otra para las categorias de
los mismos.
"""

toggle_capitales = True

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

    def add_capital_lp(self, country, id, do_fullname=False):
        capital = self.get_capital(country)
        characteristics = lt.newList(datastructure='ARRAY_LIST')
        fullname = ", ".join([capital,country])
        identifier = fullname.replace(',',"").replace("'","").replace(" ", "-").lower()
        latitude, longitude = self.capital_coords_degrees(country)
        lt.addLast(characteristics, id)
        lt.addLast(characteristics, identifier)
        lt.addLast(characteristics, fullname)
        lt.addLast(characteristics, latitude)
        lt.addLast(characteristics, longitude)
        mp.put(self.points_by_id, id, characteristics)
        if do_fullname:
            mp.put(self.points_by_name, fullname, characteristics)
        else:
            mp.put(self.points_by_name, capital, characteristics)

    def add_capitals_landing_points(self):
        country_it = ll_it.newIterator(mp.keySet(self.countries))
        i = 20000
        while ll_it.hasNext(country_it):
            country = ll_it.next(country_it)
            id = self.get_capital_id(country)
            if not id:
                self.add_capital_lp(country, i)
                i += 1
            elif country != self.id_to_country(id):
                self.add_capital_lp(country, i, do_fullname=True)
                i += 1

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

    # Clean code

    def vertex_cluster(self, vertex):
        return mp.get(self.clusters['idscc'], vertex)['value']
    
    def id_to_fullname(self, vertex):
        return lt.getElement(mp.get(self.points_by_id, vertex)['value'], 3)
    
    def id_to_city(self, vertex):
        return self.id_to_fullname(vertex).split(", ")[0]
    
    def id_to_country(self, vertex):
        return self.id_to_fullname(vertex).split(", ")[-1]
    
    def name_to_id(self, vertex):
        landing_point = mp.get(self.points_by_name, vertex)
        if landing_point:
            return lt.getElement(landing_point['value'], 1)
        else:
            return None

    def get_capital_id(self, country):
        return self.name_to_id(self.get_capital(country))
    
    def get_capital_id_safe(self, country):
        return self.name_to_id(self.get_capital_safe(country))

    def get_capital(self, country):
        return lt.getElement(mp.get(self.countries, country)['value'], 1)
    
    def get_capital_safe(self, country):
        id = self.get_capital_id(country)
        if self.id_to_country(id) != country:
            return ", ".join([self.get_capital(country), country])
        else:
            return self.get_capital(country)
    
    def point_coords_radians(self, vertex):
        landing_point = mp.get(self.points_by_id, vertex)['value']
        latitude, longitude = lt.getElement(landing_point, 4), lt.getElement(landing_point, 5)
        return latitude * pi / 180, longitude * pi / 180
    
    def point_coords_degrees(self, vertex):
        landing_point = mp.get(self.points_by_id, vertex)['value']
        return lt.getElement(landing_point, 4), lt.getElement(landing_point, 5)
    
    def capital_coords_degrees(self, country):
        capital = mp.get(self.countries, country)['value']
        return lt.getElement(capital, 2), lt.getElement(capital, 3)
    
    def get_internet_users(self, country):
        return lt.getElement(mp.get(self.countries, country)['value'], 7)
    
    def get_index_cable_name(self, index):
        return lt.getElement(lt.getElement(self.connections_list, index),3)

    # Optimizaciones de requerimientos

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
        point_it = ll_it.newIterator(mp.keySet(self.points_by_id))
        while ll_it.hasNext(point_it):
            vertex = ll_it.next(point_it)
            country = self.id_to_country(vertex)
            if country == "Colombia":
                country = country
            pair = mp.get(self.point_country, country)
            if pair:
                lt.addLast(pair['value'], vertex)
            else:
                new_list = lt.newList()
                lt.addLast(new_list, vertex)
                mp.put(self.point_country, country, new_list)
    
    def haversine(self, *args):
        if len(args) == 2:
            lat_1, lon_1 = self.point_coords_radians(args[0])
            lat_2, lon_2 = self.point_coords_radians(args[1])
        elif len(args) == 3:
            lat_1, lon_1 = args[0] * pi / 180, args[1] * pi / 180
            lat_2, lon_2 = self.point_coords_radians(args[2])
        else:
            lat_1, lon_1 = args[0] * pi / 180, args[1] * pi / 180
            lat_2, lon_2 = args[2] * pi / 180, args[3] * pi / 180
        respuesta = 12742 * asin(((sin((lat_2 - lat_1) / 2)) ** 2 + cos(lat_2) * cos(lat_1) * (sin((lon_2 - lon_1) / 2)) ** 2) ** 0.5)
        return respuesta

    def add_vertices(self):
        vertex_it = ll_it.newIterator(mp.valueSet(self.points_by_id))
        while ll_it.hasNext(vertex_it):
            new_vertex = lt.getElement(ll_it.next(vertex_it), 1)
            gp.insertVertex(self.connections_map, new_vertex)

    def add_edges(self):
        edge_it = al_it.newIterator(self.connections_list)
        while al_it.hasNext(edge_it):
            temp = al_it.next(edge_it)
            new_edge = lt.getElement(temp, 1), lt.getElement(temp, 2)
            gp.addEdge(self.connections_map, *new_edge, self.haversine(*new_edge))
    
    def add_capital_edges(self, country):
        if country == "Colombia":
            country = country
        capital = self.get_capital_id_safe(country)
        cities = mp.get(self.point_country, country)
        if cities:
            city_it = ll_it.newIterator(cities['value'])
            while ll_it.hasNext(city_it):
                city = ll_it.next(city_it)
                if capital != city:
                    if not gp.getEdge(self.connections_map, capital, city):
                        gp.addEdge(self.connections_map, capital, city, self.haversine(capital, city))
                        gp.addEdge(self.connections_map, city, capital, self.haversine(capital, city))

    def add_capitals_edges(self):
        cp = self.connections_map
        country_it = ll_it.newIterator(mp.keySet(self.countries))
        while ll_it.hasNext(country_it):
            country = ll_it.next(country_it)
            self.add_capital_edges(country)


    def draw_connections(self):
        self.connections_map = gp.newGraph(size=lt.size(self.connections_list), directed=True)
        self.add_vertices()
        self.add_edges()
        if toggle_capitales:
            self.add_capitals_edges()

    def add_country_cable(self, cable, i):
        name = lt.getElement(cable, 3)
        country = self.id_to_country(lt.getElement(cable, 1))
        if mp.get(self.cable_bandwith, name):
            countries = mp.get(self.cable_bandwith, name)['value'][1]
        else:
            bandwith = lt.getElement(cable, 8)
            countries = mp.newMap(numelements=5)
            mp.put(self.cable_bandwith, name, (bandwith, countries, i))
        mp.put(countries, country, 1)

    def cable_map_to_list(self, cable):
        cable_map = mp.get(self.cable_bandwith, cable)['value']
        cable_list = mp.keySet(cable_map[1])
        mp.put(self.cable_bandwith, cable, (cable_map[0], cable_list, cable_map[2]))

    def create_cable_map(self):
        self.cable_bandwith = mp.newMap(numelements=3268)
        cable_it = al_it.newIterator(self.connections_list)
        i = 1
        while al_it.hasNext(cable_it):
            cable = al_it.next(cable_it)
            self.add_country_cable(cable, i)
            i += 1
        cable_it = ll_it.newIterator(mp.keySet(self.cable_bandwith))
        while ll_it.hasNext(cable_it):
            cable = ll_it.next(cable_it)
            self.cable_map_to_list(cable)
        
    
    def __init__(self, filepath_points: str, filepath_connections: str, filepath_countries: str):
        (self.points_by_id, self.points_by_name, self.connections_list, self.countries,
        self.connections_map, self.point_country, self.clusters, self.cluster_number,
        self.mas_grande, self.cable_bandwith) = None, None, None, None, None, None, None, None, None, None
        self.open_countries(filepath_countries)
        self.open_points(filepath_points)
        if toggle_capitales:
            self.add_capitals_landing_points()
        self.assign_points_to_countries()
        self.open_connections(filepath_connections)
        self.draw_connections()
        self.req_1_optimization()
        self.cluster_mas_grande()
        self.create_cable_map()
    
    # Requerimiento 8

    # Requerimientos 1-7
    def req_1(self, lp1, lp2):
        id_1 = self.name_to_id(lp1)
        id_2 = self.name_to_id(lp2)
        respuesta = scc.stronglyConnected(self.clusters, id_1, id_2)
        if respuesta:
            graphing_helper(self).all_points_cluster(mp.get(self.clusters['idscc'], id_1)['value'])
        return respuesta

    def req_2_cmp_func(self, a, b):
        return a[2] > b[2]

    def req_2(self):
        respuesta = lt.newList(datastructure='ARRAY_LIST')
        vertices = gp.vertices(self.connections_map)
        vertex_it = ll_it.newIterator(vertices)
        while ll_it.hasNext(vertex_it):
            vertex = ll_it.next(vertex_it)
            name = self.id_to_fullname(vertex)
            edges = gp.degree(self.connections_map, vertex)
            lt.addLast(respuesta,[name,vertex,edges])
        mergesort.sort(respuesta, self.req_2_cmp_func)
        graphing_helper(self).graph_req_2(respuesta)
        respuesta = "\n".join(["{} ({}): {} conexiones".format(*t) for t in respuesta['elements']])
        return respuesta

    def path_str(self, path):
        respuesta = []
        path_it = ll_it.newIterator(path)
        while ll_it.hasNext(path_it):
            edge = ll_it.next(path_it)
            dist = round(edge['weight'], 2)
            cityA = self.id_to_city(edge['vertexA'])
            cityB = self.id_to_city(edge['vertexB'])
            respuesta.append("{} - {}: {} km".format(cityA,cityB,dist))
        return "\n".join(respuesta)
    
    def get_shortest_path(self, pointA, pointB):
        search_map = dij.Dijkstra(self.connections_map, pointA)
        dist = dij.distTo(search_map, pointB)
        path = dij.pathTo(search_map, pointB)
        return path, dist
    
    def req_3(self, countryA, countryB):
        pointA = self.get_capital_id_safe(countryA)
        pointB = self.get_capital_id_safe(countryB)
        path, dist = self.get_shortest_path(pointA, pointB)
        graphing_helper(self).graph_path(path)
        path = self.path_str(path)
        return path, round(dist,2)
    
    def graph_mst_prim(self, scan):
        graph = gp.newGraph(size=self.mas_grande[1])
        distTo = scan['distTo']
        edgeTo = scan ['edgeTo']
        dist_it = ll_it.newIterator(mp.keySet(distTo))
        total_dist = 0
        while ll_it.hasNext(dist_it):
            vertex = ll_it.next(dist_it)
            dist = mp.get(distTo, vertex)['value']
            if dist:
                total_dist += dist
                edge = mp.get(edgeTo, vertex)['value']
                if not gp.containsVertex(graph, edge['vertexA']):
                    gp.insertVertex(graph, edge['vertexA'])
                if not gp.containsVertex(graph, edge['vertexB']):
                    gp.insertVertex(graph, edge['vertexB'])
                gp.addEdge(graph, edge['vertexA'], edge['vertexB'], dist)
        return graph, total_dist
    
    def prim_path_to_root(self, vertex, root, edgeTo):
        path = lt.newList()
        while vertex != root:
            edge = mp.get(edgeTo, vertex)['value']
            lt.addLast(path, {'vertexA': edge['vertexB'], 'vertexB': edge['vertexA'], 'weight': edge['weight']})
            vertex = edge['vertexA']
        return path
    
    def prim_max_dist_vertex(self, graph):
        root = self.mas_grande[2]
        search = dij.Dijkstra(graph, root)
        vertex_it = ll_it.newIterator(gp.vertices(graph))
        maximum = 0, 0
        while ll_it.hasNext(vertex_it):
            vertex = ll_it.next(vertex_it)
            dist = dij.distTo(search, vertex)
            if dist != float('inf'):
                if dist > maximum[1]:
                    maximum = vertex, dist
        return maximum

    def req_4(self):
        root = self.mas_grande[2]
        search = prim.PrimMST(self.connections_map)
        scan = prim.scan(self.connections_map, search, root)
        graph, total_dist = self.graph_mst_prim(scan)
        vertex, dist = self.prim_max_dist_vertex(graph)
        path = self.prim_path_to_root(vertex, root, scan['edgeTo'])
        graphing_helper(self).graph_path(path)
        path = self.path_str(path)
        return self.mas_grande[1], path, round(dist, 2), round(total_dist, 2)
        
    def add_affected_country(self, affected_countries, id, dist1):
        country = self.id_to_country(id)
        dist2 = float('inf')
        if mp.get(affected_countries, country):
            dist2 = mp.get(affected_countries, country)['value'][1]
        mp.put(affected_countries, country, (country, min(dist1, dist2)))
    
    def req_5_cmpfunc(self, a, b):
        return a[1] > b[1]
    
    def linked_to_array(self, list):
        new_list = lt.newList(datastructure='ARRAY_LIST')
        old_it = ll_it.newIterator(list)
        while ll_it.hasNext(old_it):
            lt.addLast(new_list, ll_it.next(old_it))
        return new_list

    def req_5(self, lp):
        id = self.name_to_id(lp)
        affected_countries = mp.newMap(numelements=10)
        self.add_affected_country(affected_countries, id, 0)
        edges_it = ll_it.newIterator(gp.adjacentEdges(self.connections_map, id))
        while ll_it.hasNext(edges_it):
            edge = ll_it.next(edges_it)
            self.add_affected_country(affected_countries, edge['vertexB'], edge['weight'])
        respuesta = mp.valueSet(affected_countries)
        respuesta = self.linked_to_array(respuesta)
        mergesort.sort(respuesta, self.req_5_cmpfunc)
        graphing = graphing_helper(self)
        graphing.graph_edges_lp(id)
        graphing.graph_affected_countries(respuesta)
        return respuesta

    def req_6(self, server, cable):
        respuesta = lt.newList('ARRAY_LIST')
        bandwith, countries, index = mp.get(self.cable_bandwith, cable)['value']
        graphing = graphing_helper(self)
        graphing.graph_cable_line(index)
        country_it = ll_it.newIterator(countries)
        has_server = False
        while ll_it.hasNext(country_it):
            country = ll_it.next(country_it)
            if country != server:
                users = self.get_internet_users(country)
                guaranteed = round(bandwith * 1000000 / users, 3)
                lt.addLast(respuesta, (country, guaranteed))
            else:
                has_server = True
        graphing.graph_capacities(respuesta)
        respuesta = "\n".join(["{}: {} Mbps".format(*args) for args in respuesta['elements']])
        return respuesta, has_server
            
    def nearest_landing_point(self, IP, graphing):
        latitude, longitude = get_location(IP)
        country = ipapi.location(IP)['country_name']
        city_it = ll_it.newIterator(mp.get(self.point_country, country)['value'])
        minimum = 0, float('inf')
        while ll_it.hasNext(city_it):
            city = ll_it.next(city_it)
            dist = self.haversine(latitude, longitude, city)
            if dist < minimum[1]:
                minimum = city, dist
        graphing.graph_line_7(latitude, longitude, IP, minimum[0])
        return minimum

    def req_7(self, IP1, IP2):
        graphing = graphing_helper(self)
        vertexA, distA = self.nearest_landing_point(IP1, graphing)
        vertexB, distB = self.nearest_landing_point(IP2, graphing)
        search = dij.Dijkstra(self.connections_map, vertexA)
        path = dij.pathTo(search, vertexB)
        graphing.graph_path(path)
        path = self.path_str(path)
        appendA = "{} - {}: {} km\n".format(IP1, self.id_to_city(vertexA), round(distA, 2))
        appendB = "\n{} - {}: {} km".format(self.id_to_city(vertexB), IP2, round(distB ,2))
        path = appendA + path + appendB
        dist = dij.distTo(search, vertexB) + distA + distB
        return path, round(dist, 2)

class graphing_helper:
    def graph_point(self, point):
        coordinates = self.data.point_coords_degrees(point)
        name = self.data.id_to_fullname(point)
        folium.Marker(coordinates, popup=name).add_to(self.m)
        lt.addLast(self.line, coordinates)
    
    def graph_line_7(self, latitude, longitude, name, point):
        coordinates = self.data.point_coords_degrees(point)
        folium.PolyLine([(latitude, longitude), coordinates]).add_to(self.m)
        folium.Marker([latitude, longitude], popup=name).add_to(self.m)

    def graph_path(self, path):
        first = True
        path_it = ll_it.newIterator(path)
        while ll_it.hasNext(path_it):
            edge = ll_it.next(path_it)
            if first:
                self.graph_point(edge['vertexA'])
                first = False
            self.graph_point(edge['vertexB'])
        folium.PolyLine(self.line['elements']).add_to(self.m)
        self.m.save('Data/map.html')
    
    def graph_req_2(self, conexion_lp):
        conexion_it = al_it.newIterator(conexion_lp)
        while al_it.hasNext(conexion_it):
            name, vertex, edges = tuple(al_it.next(conexion_it))
            coordinates = self.data.point_coords_degrees(vertex)
            folium.Circle(coordinates, radius=(edges**(0.5)*9000), popup="{}: {}".format(name, edges), fill_color="#3186cc").add_to(self.m)
        self.m.save('Data/map.html')
    
    def graph_line_6(self, index):
        connection = lt.getElement(self.data.connections_list, index)
        pointA = lt.getElement(connection, 1)
        pointB = lt.getElement(connection, 2)
        coordsA = self.data.point_coords_degrees(pointA)
        coordsB = self.data. point_coords_degrees(pointB)
        folium.PolyLine([coordsA, coordsB]).add_to(self.m)

    def graph_cable_line(self, index):
        i = index
        cable_name = self.data.get_index_cable_name(index)
        while self.data.get_index_cable_name(i) == cable_name:
            self.graph_line_6(i)
            i += 1

    def graph_capacities(self, capacities):
        capacity_it = al_it.newIterator(capacities)
        while al_it.hasNext(capacity_it):
            country, capacity = al_it.next(capacity_it)
            capital = self.data.get_capital_id_safe(country)
            coords = self.data.point_coords_degrees(capital)
            folium.Marker(coords, popup="{}: {} Mbps".format(country, capacity)).add_to(self.m)
        self.m.save('Data/map.html')

    def graph_edge(self, edge):
        coordsA = self.data.point_coords_degrees(edge['vertexA'])
        coordsB = self.data.point_coords_degrees(edge['vertexB'])
        folium.PolyLine([coordsA, coordsB]).add_to(self.m)

    def graph_edges_lp(self, lp):
        edges_it = ll_it.newIterator(gp.adjacentEdges(self.data.connections_map, lp))
        while ll_it.hasNext(edges_it):
            edge = ll_it.next(edges_it)
            self.graph_edge(edge)

    def graph_affected_countries(self, countries):
        country_it = al_it.newIterator(countries)
        while al_it.hasNext(country_it):
            country, dist = al_it.next(country_it)
            capital = self.data.get_capital_id_safe(country)
            coords = self.data.point_coords_degrees(capital)
            folium.Marker(coords, popup="{}: {} kms".format(country, dist)).add_to(self.m)
        self.m.save('Data/map.html')
    
    def all_points_cluster(self, cluster):
        vertex_it = ll_it.newIterator(mp.keySet(self.data.clusters['idscc']))
        while ll_it.hasNext(vertex_it):
            vertex = ll_it.next(vertex_it)
            if mp.get(self.data.clusters['idscc'], vertex)['value'] == cluster:
                self.graph_point(vertex)
        self.m.save('Data/map.html')

    def __init__(self, data):
        self.line = lt.newList(datastructure='ARRAY_LIST')
        self.m = folium.Map()
        self.data = data
        self.data: landing_points
    

# Funciones para agregar informacion al catalogo

# Funciones de consulta

# Funciones utilizadas para comparar elementos dentro de una lista

# Funciones de ordenamiento

if __name__ == '__main__':
    ld = landing_points("Data/landing_points.csv", "Data/connections.csv","Data/countries.csv")
    print(ld.req_1("Redondo Beach", "Vung Tau"))