import os
from datetime import datetime
import random

import scipy.cluster.hierarchy as sch
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Ellipse
import pandas as pd

import xml_parsing as xp


class OptimalVisualization:
    # this class deals with generating the optimal visualization (zoom) of two dendrograms

    NB_ELEMENTS = 0

    def __init__(self, models, alpha, NB_ELEMENTS):
        self.models = models
        self.alpha = alpha
        self.NB_ELEMENTS = NB_ELEMENTS

    def calculate_scores(self):

        models = self.models
        alpha = self.alpha

        distance_matrices = [[], []]
        granularity_list = [[], []]


        for i in range(2):
            # print('For model: {0} :'.format(i))
            for x in range(1, self.NB_ELEMENTS + 1):
                d = sch.dendrogram(xp.get_linkage_matrix(models[i]), orientation='left', p=x, truncate_mode='lastp',
                                   no_plot=True)
                dist_matrix = np.array(self._get_distance_matrix(x, models[i], d))
                distance_matrices[i].append(dist_matrix)
                # print('at level: ', x)
                # print('distance matrix: \n', np.matrix(xp.make_sym_matrix(dist_matrix)))
                # print('granularity score: \n', self._granularity_score(d))
                granularity_list[i].append(self._granularity_score(d))
                # print('********************************')
                # df = pd.DataFrame(dist_matrix)
                # df.to_excel(writers[i], sheet_name='dendrogram #' + str(x))
            # writers[i].save()


        similarity_matrix, granularity_matrix = self.similarity_score(distance_matrices,
                                                                          granularity_list)  # getting the similarity of the distance matrices

        # print('Similarity Matrix: \n', similarity_matrix)
        # print('Granularity Matrix: \n', granularity_matrix)
        overall_sim = np.zeros((self.NB_ELEMENTS, self.NB_ELEMENTS))
        # print('alpha: ', alpha)
        for i in range(self.NB_ELEMENTS):
            for j in range(self.NB_ELEMENTS):
                overall_sim[i][j] = ((alpha * similarity_matrix[i][j]) + ((1 - alpha) * granularity_matrix[i][j]))
        # print('Overall Matrix : \n', overall_sim)

        # file_directory = 'excel/' + str(datetime.now().date()).replace('-', '') + '_' + str(datetime.now().hour) + str(
        #     datetime.now().minute) + '/'
        # if not os.path.exists(file_directory):
        #     os.makedirs(file_directory)
        # writer = pd.ExcelWriter(os.path.join(file_directory, 'similarity_granularity.xlsx'), engine='xlsxwriter')
        # df = pd.DataFrame(granularity_matrix)
        # df.to_excel(writer, sheet_name='granularity')
        # df = pd.DataFrame(similarity_matrix)
        # df.to_excel(writer, sheet_name='similarity')
        # writer.save()

        max_val = 0
        highest_i = self.NB_ELEMENTS
        highest_j = self.NB_ELEMENTS
        for i in range(self.NB_ELEMENTS):
            for j in range(self.NB_ELEMENTS):
                if overall_sim[i][j] > max_val:
                    max_val = overall_sim[i][j]
                    highest_i = i
                    highest_j = j

        # print('highest_i: {} , highest_j: {}'.format(highest_i, highest_j))
        return ((highest_i+1), (highest_j+1))

    def similarity_score(self, distance_matrices, granularity_list):
        sim_matrix = np.zeros((len(distance_matrices[0]), len(distance_matrices[0])))
        info_matrix = np.zeros((len(granularity_list[0]), len(granularity_list[0])))
        maxval = 0
        max_i = -1
        max_j = -1

        #print distance matrix


        for i in range(len(distance_matrices[0])):
            #for j in range(i + 1):
            for j in range(len(distance_matrices[0])):
                if i != 0 or j != 0:
                    sim = self._manhattan_modified(distance_matrices[0][i], distance_matrices[1][j])
                    # print('sim of ({},{}) = {}'.format(i,j,sim))
                    sim_matrix[i][j] = sim
                    if sim > maxval:
                        maxval = sim
                        max_i = i
                        max_j = j
                    info_matrix[i][j] = granularity_list[0][i] * granularity_list[1][j]
        sim_matrix[0][0] = 0
        #sim_matrix = xp.make_sym_matrix(sim_matrix)
        info_matrix = xp.make_sym_matrix(info_matrix)

        # print('max value of similarity = {0}, of [{1}][{2}]'.format(maxval, max_i, max_j))
        return sim_matrix, info_matrix

    def _granularity_score(self, dend):
        shown_nodes = len(dend['leaves'])
        all_nodes = self.NB_ELEMENTS

        informativeness = (shown_nodes - 1) / (all_nodes - 1)
        granularity = 1- informativeness
        return granularity

    def _manhattan_modified(self, m1, m2):
        n = len(m1)

        sum1 = 0
        sum2 = 0

        # print('*************************************')
        # print('m1: ', m1)
        # print('m2: ', m2)

        for i in range(n):
            for j in range(n):
                if i != 0 or j != 0:
                    sum1 += abs(m1[i][j] - m2[i][j])
                    sum2 += m1[i][j] + m2[i][j]
                    # print(
                    #     'at i: {0}, j: {1},, sum1 = {2} and sum2 = {3} after adding {4} & {5}'.format(i, j, sum1, sum2,
                    #                                                                                   m1[i][j],
                    #                                                                                   m2[i][j]))

        dist = sum1 / sum2
        # print('dist: ', dist)
        result = 1 - dist  # similarity
        # print('result: ', result)
        # print('*****************************************************')
        return result

    def _get_distance_matrix(self, nb_nodes, model, dend):
        # We should be doing this for every zooming level dendrogram (dend)
        # nb_nodes corresponds to the zooming level

        dist_matrix = [[0 for a in range(self.NB_ELEMENTS)] for b in range(self.NB_ELEMENTS)]
        for i in range(len(dist_matrix)):
            for j in range(0, i):
                dist_matrix[i][j] = self._get_height(i, j, nb_nodes, dend, model)
            # print("height for [{0}][{1}] = {2}".format(i,j,dist_matrix[i][j]))
        return dist_matrix

    def _get_height(self, i, j, nb_nodes, dend, model, type='our approach'):
        # 1- get the clusters
        # 2- get the top most n clusters (n is the number of shown nodes/ zooming level) as visible clusters
        #       2.1- group the hidden clusters that share a common node
        # 3- get heights according to centers
        # 4- done

        clusters = DendrogramTools.get_basic_components(
            model, self.NB_ELEMENTS)  # get the basic components that form clusters as lists of lists
        visible_clusters = clusters[-(nb_nodes - 1):]  # n-1 by trial and error on paper

        # print('visible clusters: ', visible_clusters)
        hidden_clusters_list = clusters[:len(clusters) - len(visible_clusters)]
        for ind, hc in enumerate(hidden_clusters_list):
            hidden_clusters_list[ind] = set(hidden_clusters_list[ind])

        heights = [a[0] for a in DendrogramTools.get_dend_centers(dend)]

        #print('Heights array: ', heights)

        if (type == 'our approach'):
            if self._same_hidden_cluster(i, j, hidden_clusters_list) or (len(heights) == 0):
                return 0
            # convention used: height of the least common node
            else:  # if two nodes are not of the same cluster we get the height of the lowest common node
                for ind, c in enumerate(visible_clusters):
                    if i in c and j in c:
                        return heights[ind]

    def _same_hidden_cluster(self, i, j, hidden_cluster_list):
        for hc in hidden_cluster_list:
            if i in hc and j in hc:
                return True

        return False


class LinksGeneration:
    @staticmethod
    def connect_components(basic_components, list_of_centers, ax1, ax2, patch_color,
                           status='exact', threshold=0.5, number_of_elements_hide_link=0):


        """
        :param number_of_elements_hide_link: `n`; if the node has e <= n => the link will be replaced by dots
        :param basic_components: the set of sets which the dendrogram is formed of
        :param list_of_centers: array of size 2, where each array stores the locations of the centers of the
                                dendrogram's archs
        :param ax1: axis used for dendrogram #1
        :param ax2: axis used for dendrogram #2
        :param patch_color: color of the link
        :param status: when to connect: exact / different
        :param threshold: if different is selected, the threshold in which we draw links
        :param ylim: the highest value possible on the y-axis
        :param xlim: the highest value possible on the x-axis
        :return:
        """
        # print('Generating new links ....')
        connection_patches = [[],[]]
        circles = [[], []]
        # displayed components (excluding the ones hidden with zooming)
        basic_components = [basic_components[0][-len(list_of_centers[0]):],
                            basic_components[1][-len(list_of_centers[1]):]]

        jaccard_matrix, avg = LinksGeneration._generate_jaccard_sim_matrix(basic_components)

        # file_directory = 'excel/' + str(datetime.now().date()).replace('-', '') + '_' + str(datetime.now().hour) + str(
        #     datetime.now().minute) + '/'
        # if not os.path.exists(file_directory):
        #     os.makedirs(file_directory)
        # writer = pd.ExcelWriter(os.path.join(file_directory, 'transporation.xlsx'), engine='xlsxwriter')
        # df = pd.DataFrame(jaccard_matrix)
        # df.to_excel(writer, sheet_name='zooming(' + str(len(jaccard_matrix)) +',' + str(len(jaccard_matrix[0]))+')')

        # writer.save()


        connections = LinksGeneration._generate_connections_from_matrix(jaccard_matrix,
                                                                        threshold=threshold if status != 'exact' else 0.5)
        #    print("Connections: \n", connections)
        colors_to_hex = {'Red': '#FF0000',
                         'Green': '#008000',
                         'Blue': '#0000FF',
                         'Cyan': '#00FFFF',
                         'Magenta': '#FF00FF',
                         'Yellow': '#FFFF00',
                         'Black': '#696969'
                         }

        ax1_xlim = ax1.get_xlim()[0]
        ax1_ylim = ax1.get_ylim()[1]

        ax2_xlim = ax2.get_xlim()[1]
        ax2_ylim = ax2.get_ylim()[1]

        for ind, c in enumerate(connections):
            node_order_in_cluster_1 = c[0][0]
            node_order_in_cluster_2 = c[0][1]
            connection_strength = c[1]
            # print('connection_strength: ', c[1])
            connection_patch = \
                ConnectionPatch(xyA=list_of_centers[0][node_order_in_cluster_1],
                                xyB=list_of_centers[1][node_order_in_cluster_2],
                                coordsA='data', coordsB='data',
                                axesA=ax1, axesB=ax2, arrowstyle='-', clip_on=False,
                                lw=connection_strength*2.5, connectionstyle='arc3, rad=0.1',
                                color=LinksGeneration._color_variant(colors_to_hex[patch_color],
                                                                     connection_strength * (-120)),
                                zorder=9)

            if len(basic_components[0][node_order_in_cluster_1]) > number_of_elements_hide_link:
                connection_patches[1].append((connection_patch,
                                              (basic_components[0][node_order_in_cluster_1],
                                               basic_components[1][node_order_in_cluster_2])
                                              ))

            else:
                connection_patches[0].append((connection_patch,
                                              (basic_components[0][node_order_in_cluster_1],
                                               basic_components[1][node_order_in_cluster_2])
                                              ))
                radius = 0.01
                equivalent_radii = [LinksGeneration._radius_conversion(ax1_xlim, ax1_ylim, radius),
                                    LinksGeneration._radius_conversion(ax2_xlim, ax2_ylim, radius)]
                color = LinksGeneration._generate_hex_color()
                circles[0].append((
                    Ellipse((list_of_centers[0][node_order_in_cluster_1]), radius, equivalent_radii[0], color=color,
                            zorder=10), basic_components[0][node_order_in_cluster_1]))

                circles[1].append((
                    Ellipse((list_of_centers[1][node_order_in_cluster_2]), radius, equivalent_radii[1], color=color,
                            zorder=10), basic_components[1][node_order_in_cluster_2]))

        # print('circles generated: ', circles)
        return connection_patches, circles, avg

    @staticmethod
    def _generate_hex_color():
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0, 255)
        to_return = '#%02x%02x%02x' % (r, g, b)
        # print('color generated is: ', to_return)
        return to_return

    @staticmethod
    def _radius_conversion(xlim, ylim, radius):
        """
        :param xlim: the highest value displayed on the x-axis
        :param ylim: the highest value displayed on the x-axis
        :param radius: the radius of the circle
        :return: the height of the ellipse to make it look like a circle (as x-axis and y-axis have different scales)
        """
        return ylim * radius / xlim

    @staticmethod
    def _color_variant(hex_color, brightness_offset=1):
        '''takes a color like #87c95f and produces a lighter or darker variant '''
        if len(hex_color) != 7:
            raise Exception("Passed %s into _color_variant(), needs to be in #87c95f format." % hex_color)
        rgb_hex = [hex_color[x:x + 2] for x in [1, 3, 5]]
        new_rgb_int = [int(hex_value, 16) + brightness_offset for hex_value in rgb_hex]
        # print('new rgb int:', new_rgb_int)
        for k in range(len(new_rgb_int)):
            # print(k)
            # print(new_rgb_int)
            a = max(0, new_rgb_int[k])
            new_rgb_int[k] = min(255, a)  # make sure new values are between 0 and 255
        to_return = '#%02x%02x%02x' % (int(new_rgb_int[0]), int(new_rgb_int[1]), int(new_rgb_int[2]))
        # print(to_return)
        return to_return

    @staticmethod
    def _generate_connections_from_matrix(matrix, threshold=0):
        """
        :param matrix: matrix(i,j) is the jaccard distance between nodes i & j
        :param threshold:
        :return: a list where each element is a tuple: (list of nodes that should be connected,
                how strong their connection is)
        """
        eliminated_i = []
        eliminated_j = []
        connected = []
        for iteration in range(min(len(matrix), len(matrix[0]))):
            max_val = -1
            potential_i = -1
            potential_j = -1
            for i in range(len(matrix)):
                if i not in eliminated_i:
                    for j in range(len(matrix[0])):
                        if j not in eliminated_j:
                            if matrix[i][j] > max_val:
                                max_val = matrix[i][j]
                                potential_i = i
                                potential_j = j
            if max_val >= threshold:
                eliminated_i.append(potential_i)
                eliminated_j.append(potential_j)
                connected.append(([potential_i, potential_j], max_val))
        # print('connected: ', connected)
        return connected

    @staticmethod
    def _generate_jaccard_sim_matrix(components):
        # input is a matrix of 2 vectors, each vector i is the list of nodes that dendrogram i consists of
        matrix = np.zeros((len(components[0]), len(components[1])))
        sum = 0
        for i in range(len(components[0])):
            for j in range(len(components[1])):
                matrix[i][j] = LinksGeneration._jaccard_similarity(components[0][i], components[1][j])
                sum += matrix[i][j]
                # print('matrix[{},{}] = {}'.format(i,j,matrix[i][j]))
                # print('sum of jacc is: ', sum)

        avg = sum / (len(components[0]) * len(components[1]))
        # print('Average is: ', avg)
        return matrix, avg

    @staticmethod
    def _jaccard_similarity(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union


class DendrogramTools:

    @staticmethod
    def get_basic_components(model, NB_ELEMENTS):
        temp_dict = {}
        cluster_combinations = model.children_

        to_return = []
        MAX_INDEX = NB_ELEMENTS - 1
        count = NB_ELEMENTS

        # corner case: more than 2 in the same cluster
        for i in range(len(cluster_combinations)):
            temp = []
            for j in range(2):
                x = cluster_combinations[i][j]
                if x <= MAX_INDEX:
                    temp.append(x)
                else:
                    a = DendrogramTools.grab_deep_elements(x, temp_dict, MAX_INDEX)
                    for el in a:
                        temp.append(el)
            temp_dict[count] = temp
            count += 1
            to_return.append(temp)

        return to_return

    @staticmethod
    def grab_deep_elements(x, dict, MAX_INDEX):
        lista = dict.get(x)
        returned_list = []

        for el in lista:
            if el <= MAX_INDEX:
                returned_list.append(el)
            else:
                DendrogramTools.grab_deep_elements(el, dict, MAX_INDEX)

        return returned_list

    @staticmethod
    def get_dend_centers(d):
        dend_centers = []
        for ycoords, xcoords in zip(d['icoord'], d['dcoord']):
            #   To get the center: x= f (or g), y = (a+b)/2
            xcenter = xcoords[1]
            ycenter = (ycoords[2] + ycoords[1]) / 2
            dend_centers.append((xcenter, ycenter))
        dend_centers.sort(key= lambda x:x[0])
        #print('Dendrogram centers: ', dend_centers)
        return dend_centers
