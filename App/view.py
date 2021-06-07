"""
 * Copyright 2020, Departamento de sistemas y Computación, Universidad
 * de Los Andes
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

import os
import config as cf
import sys
import webbrowser
import App.controller as controller
from DISClib.ADT import list as lt
from DISClib.DataStructures import arraylistiterator as ll_it
assert cf


"""
La vista se encarga de la interacción con el usuario
Presenta el menu de opciones y por cada seleccion
se hace la solicitud al controlador para ejecutar la
operación solicitada
"""

def printMenu():
    print("Bienvenido")
    print("1- Cargar información en el catálogo")
    print("2- Req_1: Componentes conectados")
    print("3- Req_2: Infraestructura Crítica")
    print("4- Req_3: Ruta entre capitales")
    print("5- Req_4: Red de Expansión Mínima")
    print("6- Req_5: Impacto de fallo en landing point")
    print("7- Req_6: Ancho de banda máximo para transmisión")
    print("8- Req_7: Ruta entre IPs")


catalog = None

"""
Menu principal
"""
while True:
    printMenu()
    inputs = input('Seleccione una opción para continuar\n')
    if int(inputs[0]) == 1:
        print("Cargando información de los archivos ....")
        data, landing_points, connections, countries = controller.cargar_datos()
        print("Hay {} landing points, {} conexiones bilaterales1 entre estos y {} paises.".format(landing_points, connections, countries))
    elif int(inputs[0]) == 2:
        lp1 = input("Introduzca el nombre del primer landing point: ")
        lp2 = input("Introduzca el nombre del segundo landing point: ")
        conectados = controller.req_1(data, lp1, lp2)
        print("Hay {} clusters.".format(data.cluster_number))
        if conectados:
            print("{} y {} están conectados.".format(lp1, lp2))
            input("Presione alguna tecla para ver la visualización ")
            webbrowser.open('file://' + os.path.realpath('Data/map.html'))
        else:
            print("{} y {} no están conectados".format(lp1, lp2))
    elif int(inputs[0]) == 3:
        x = int(input("Introduzca el número de landing points que quiere que aparezcan en la consola: "))
        tabla = controller.req_2(data)
        print("\n".join(tabla.split("\n")[:x]))
        input("Presione alguna tecla para ver la visualización ")
        webbrowser.open('file://' + os.path.realpath('Data/map.html'))
    elif int(inputs[0]) == 4:
        countryA = input("Introduzca el primer país: ")
        countryB = input("Introduzca el otro país: ")
        trayecto, distancia = controller.req_3(data, countryA, countryB)
        print(trayecto)
        print("La distancia total del trayecto es {} kms.".format(distancia))
        input("Presione alguna tecla para ver la visualización ")
        webbrowser.open('file://' + os.path.realpath('Data/map.html'))
    elif int(inputs[0]) == 5:
        nodos, rama, distancia_rama, distancia_total = controller.req_4(data)
        print("La red de expansión mínima más grande tiene {} nodos.".format(nodos))
        print("La rama más grande de dicha red tiene {} kms y es:".format(distancia_rama))
        print(rama)
        print("La red cubre {} kms.".format(distancia_total))
        input("Presione alguna tecla para ver la visualización ")
        webbrowser.open('file://' + os.path.realpath('Data/map.html'))
    elif int(inputs[0]) == 6:
        lp = input("Introduzca el landing point: ")
        lista_paises = controller.req_5(data, lp)
        print("Hay {} paises afectados".format(lt.size(lista_paises)))
        iterador = ll_it.newIterator(lista_paises)
        while ll_it.hasNext(iterador):
            tupla = ll_it.next(iterador)
            print("{} es afectado. Su landing point más cercano está a {} kms de {}.".format(tupla[0], round(tupla[1], 2), lp))
        input("Presione alguna tecla para ver la visualización ")
        webbrowser.open('file://' + os.path.realpath('Data/map.html'))
    elif int(inputs[0]) == 7:
        server = input("Introduzca el país donde se ubica el servidor: ")
        cable = input("Introduzca el nombre del cable: ")
        tabla = controller.req_6(data, cable, server)
        print(tabla[0])
        input("Presione alguna tecla para ver la visualización ")
        webbrowser.open('file://' + os.path.realpath('Data/map.html'))
    elif int(inputs[0]) == 8:
        IP1 = input("Introduzca la primera IP: ")
        IP2 = input("Introduzca la seguna IP: ")
        trayecto, distancia = controller.req_7(data, IP1, IP2)
        print("La distancia total entre ambas IPs es {} kms.".format(distancia))
        print(trayecto)
        input("Presione alguna tecla para ver la visualización ")
        webbrowser.open('file://' + os.path.realpath('Data/map.html'))
    else:
        sys.exit(0)
sys.exit(0)
