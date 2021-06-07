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
 """


import config as cf
import App.model as model
from DISClib.ADT import graph as gp
from DISClib.ADT import map as mp

"""
El controlador se encarga de mediar entre la vista y el modelo.
"""

# Funciones para la carga de datos

def cargar_datos():
    data = model.landing_points("Data/landing_points.csv", "Data/connections.csv","Data/countries.csv")
    landing_points = gp.numVertices(data.connections_map)
    connections = gp.numEdges(data.connections_map) // 2
    countries = mp.size(data.countries)
    return data, landing_points, connections, countries

# Requerimientos

def req_1(data: model.landing_points, lp1, lp2):
    return data.req_1(lp1, lp2)

def req_2(data: model.landing_points):
    return data.req_2()

def req_3(data: model.landing_points, countryA, countryB):
    return data.req_3(countryA, countryB)

def req_4(data: model.landing_points):
    return data.req_4()

def req_5(data: model.landing_points, lp):
    return data.req_5(lp)

def req_6(data: model.landing_points, cable, server):
    return data.req_6(server, cable)

def req_7(data: model.landing_points, IP1, IP2):
    return data.req_7(IP1, IP2)