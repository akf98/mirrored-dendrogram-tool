import xml.etree.ElementTree as ET
import random

import numpy as np
import os

import operator as op
from functools import reduce

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

default_subweights = []
default_weights = []
NB_ELEMENTS = 10
list_of_patients = []
base_root = os.getcwd()


def set_subweights(f):
    global default_subweights, default_weights

    titles = get_section_names(f)

    default_weights = [1 / len(titles) for a in range(len(titles))]
    default_subweights = [[] for a in range(7 if f.upper() == 'C' else 8)]

    subtitles = get_subsection_names(f)
    for i in range(len(default_subweights)):
        leng = len(subtitles[i])
        default_subweights[i] = [1 / leng for a in range(leng)]
    

def get_section_names(t, original=False):
    tree = ET.parse(os.path.join(base_root,'data','{0}1.xml'.format(t.upper())))
    root = tree.getroot()
    labels = []
    for node in list(root):
        labels.append(node.tag.replace('_', ' ') if not original else node.tag)

    return labels


def get_subsection_names(t, original=False):
    tree = ET.parse(os.path.join(base_root, 'data', '{0}1.xml'.format(t.upper())))
    root = tree.getroot()
    l = []
    a = []
    for i in range(len(list(root))):
        a = []
        for subnode in list(root[i]):
            a.append(subnode.tag.replace('_', ' ') if not original else subnode.tag)
        l.append(a)
    return l


def dob_similarity(a, b, c):
    a = float(a)
    b = float(b)
    c = float(c)

    a += 249091200
    b += 249091200
    c += 249091200

    x = 1 - (abs(a - b) / abs(c))

    return x


def multivalue_similarity(a, b):
    arr1 = a.split(', ')
    arr2 = b.split(', ')

    max_val = max(len(arr1), len(arr2))

    t = list(set(arr1) & set(arr2))
    return (len(t) / max_val)


def numeric_similarity(a, b, c):
    if a == 0 and b == 0:
        x = 1

    else:
        x = 1 - (abs(a - b) / abs(c))
    return x


def text_similarity(a, b):
    if a == b:
        return 1
    elif (a == 'Cigarettes' or a == 'Hubble Bubble') and (b == 'Cigarettes' or b == 'Hubble Bubble'):
        return 0.5
    else:
        return 0


def calc_sim(patient_type_initial, patient_1_number, patient_2_number, weights=None, subWeights=None):
    global default_subweights, default_weights

    set_subweights(patient_type_initial)
    if subWeights == None:
        subWeights = default_subweights

    if weights == None:
        weights = default_weights

    tree1 = ET.parse(os.path.join(base_root, 'data','{0}{1}.xml'.format(patient_type_initial.upper(), patient_1_number)))
    tree2 = ET.parse(os.path.join(base_root, 'data','{0}{1}.xml'.format(patient_type_initial.upper(), patient_2_number)))
    tree_max = ET.parse(os.path.join(base_root, 'data','{0}{1}.xml'.format(patient_type_initial.upper(), 'Max')))

    root1 = tree1.getroot()
    root2 = tree2.getroot()
    root_max = tree_max.getroot()

    sim = []
    ind = 0
    raw_similarity_values = [[] for a in range(len(get_section_names(patient_type_initial)))]

    for child1, child2, childMax in zip(root1, root2, root_max):
        l2count = 0
        l2sim = 0
        l2sim_det = 0
        # raw_similarity_values.append([])
        for e1, e2, eMax in zip(child1, child2, childMax):

            l2count += 1

            s1 = e1.text
            s2 = e2.text
            s_max = eMax.text

            if s1 is None or s2 is None or s1 == ' ' or s2 == ' ':
                continue

            s1 = s1[1:] if s1.startswith('<') or s1.startswith('>') else s1
            s2 = s2[1:] if s2.startswith('<') or s2.startswith('>') else s2


            if e1.tag == 'DOB':
                z = dob_similarity(s1, s2, s_max)
                l2sim += z
                l2sim_det = z
            elif s1.replace('.', '', 1).isdigit() and s2.replace('.', '', 1).isdigit():
                x = numeric_similarity(float(s1), float(s2), float(s_max))
                l2sim += x
                l2sim_det = x

            elif e1.tag == 'Triggers' or e1.tag == 'Type_of_Abortive_Treatment_':
                q = multivalue_similarity(s1, s2)
                l2sim += q
                l2sim_det = q

            else:
                y = text_similarity(s1, s2)
                l2sim += y
                l2sim_det = y


            raw_similarity_values[ind].append(l2sim_det)
        ind += 1
        sim.append(l2sim / l2count)

    det_weighted_sum = 0

    normalized_weights = normalize(weights, subWeights)


    for x in range(len(raw_similarity_values)):
        for y in range(len(raw_similarity_values[x])):
            det_weighted_sum += normalized_weights[x][y] * raw_similarity_values[x][y]

    return det_weighted_sum


def normalize(weights, subWeights):
    normalized = [[] for a in range(len(subWeights))]
    for k in range(len(normalized)):
        normalized[k] = [0 for a in range(len(subWeights[k]))]
    for k in range(len(weights)):
        for i in range(len(subWeights[k])):
            s = subWeights[k][i]
            normalized[k][i] = weights[k] * subWeights[k][i]

    return normalized


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return int(numer / denom)

def get_random_patients():
    files_num = []
    for i in range(NB_ELEMENTS):
       files_num.append(random.randint(1, 114))
    return files_num

random_patients = get_random_patients()


def create_patient_distance_matrix(f, weights=None, subweights=None):
    # n is the number of files out there
    # m is the distance matrix
    global list_of_patients
    files_num = [1+a for a in range(NB_ELEMENTS)]

    n = len(files_num)

    # print('filesNum: ', files_num)
    m = np.zeros([n, n])
    for i in range(n):
        for j in range(0, i):
            m[i, j] = 1 - round(calc_sim(f, files_num[i], files_num[j], weights, subweights), 6)

    to_return = make_sym_matrix(m)
    return to_return, n


def make_sym_matrix(a):
    xs, ys = np.triu_indices(len(a))
    a[xs, ys] = a[ys, xs]
    return a


def get_linkage_matrix(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    labels_for_here = [str(a + 1) for a in range(5)]
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    return linkage_matrix


def fix_linkage_matrix(linkage):
    # if two or more clusters have the same distance, add a multiple of 0.0001 to each
    increment_value = 0.000001
    list_of_heights = []
    for row in linkage:
        list_of_heights.append(row[2])
    set_of_heights = set(list_of_heights)
    if len(list_of_heights) == len(set_of_heights):
        return linkage
    else:
        value_indeces = [[], []]  # value, indeces
        for i in range(len(list_of_heights)):
            ind = search(value_indeces[0], list_of_heights[i])
            if (ind == -1):
                value_indeces[0].append(list_of_heights[i])
                value_indeces[1].append([i])
            else:
                value_indeces[1][ind].append(i)

    duplicate_indeces = []
    for i in range(len(value_indeces[1])):
        x = value_indeces[1][i]
        if len(x) > 1:
            duplicate_indeces.append(value_indeces[1][i])

    for list_duplicates in duplicate_indeces:
        k = 0
        for k, index in enumerate(list_duplicates):
            linkage[index][2] += k * increment_value

    return linkage


def search(list, value):
    for i in range(len(list)):
        if list[i] == value:
            return i
    return -1


def get_model_variants(X, Y):
    models = []


    models.append(AgglomerativeClustering(affinity='precomputed', distance_threshold=0, linkage='average', memory=None,
                                          n_clusters=None).fit(X))

    models.append(AgglomerativeClustering(affinity='precomputed', distance_threshold=0, linkage='average', memory=None,
                                          n_clusters=None).fit(Y))

    return models


def get_parameter_p(model, zoom_percentage):
    return int(model.n_leaves_ * (zoom_percentage / 100))


def plot_dends(models, zoom_percentage1=100, zoom_percentage2=100):
    plt.close()
    plt.clf()

    plt.title('Hierarchical Clustering Dendrogram')

    plt.subplot(121)
    dendrogram(get_linkage_matrix(models[0]), orientation='left',
               p=get_parameter_p(models[0], zoom_percentage1), truncate_mode='lastp')
    plt.title('Dendrogram for 1st criteria')
    plt.subplot(122)
    dendrogram(get_linkage_matrix(models[1]), orientation='right',
               p=get_parameter_p(models[1], zoom_percentage2), truncate_mode='lastp')
    plt.title('Dendrogram for 2nd criteria')

    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


if __name__ == '__main__':
    calc_sim('C', 1, 2)

