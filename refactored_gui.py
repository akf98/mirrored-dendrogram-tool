import asyncio
import ctypes
import os
import platform
import subprocess
import sys
import time
import tkinter as tk
import xml.etree.ElementTree as et
from datetime import datetime
from tkinter import messagebox
from xml.etree import ElementTree

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import set_link_color_palette
# noinspection PyUnresolvedReferences
from xml.dom.minidom import parseString

from src.code import xmlParsing as xp

from src.code.refactoring.HelpingTools import LinksGeneration
from src.code.refactoring.HelpingTools import OptimalVisualization
from src.code.refactoring.HelpingTools import DendrogramTools

import pandas as pd
import numpy as np

from PIL import ImageTk, Image

patient_type = ''

dend_index = 0
sim_matrices = []
maxC = 83
maxM = 114
subs = []
expState = []
fixState = []
subCheckList = []
mainPositions = []
mainSubPositions = []
subWeightsBox = []
subWeights = []
threshold_value = 0
min_nodes_threshold = 0
percentage_nodes_threshold = 0
iteration = 0
hovering_on = 1
tracking_label = None
confirm_button = None

zoom_vals = []

saveButton = None
subRefreshButton = None

subFixButtons = []
subFixState = []

newWindow = None

NB_ELEMENTS = 10
nb_links = NB_ELEMENTS - 1

main_weights = [[], []]
alpha_g = 0.8

final_weights = [[[], []], [[], []]]

base_root = 'C:\\Users\\Angy\\Desktop\\'



def restart_program(root):
    root.destroy()
    python = sys.executable
    script = os.path.realpath(__file__)
    subprocess.Popen([python, script])


def set_global_vals(v):
    global expState
    global fixState
    global subs
    global subCheckList
    global mainPositions
    global mainSubPositions
    global subWeightsBox
    global subWeights
    global subFixButtons
    global subFixState

    subCheckList = [[] for a in range(7 if str(v.get()) == 'C' else 8)]
    subs = [[] for a in range(7 if str(v.get()) == 'C' else 8)]
    expState = [0 for a in range(7 if str(v.get()) == 'C' else 8)]
    fixState = [0 for a in range(7 if str(v.get()) == 'C' else 8)]
    mainSubPositions = [[] for a in range(7 if str(v.get()) == 'C' else 8)]
    mainPositions = [6 + a for a in range(7 if str(v.get()) == 'C' else 8)]
    subWeightsBox = [[] for a in range(7 if str(v.get()) == 'C' else 8)]
    subWeights = [[] for a in range(7 if str(v.get()) == 'C' else 8)]
    subFixButtons = [[] for a in range(7 if str(v.get()) == 'C' else 8)]
    subFixState = [[] for a in range(7 if str(v.get()) == 'C' else 8)]


def disableFirst(b1, b2):
    # cb1['state']='disabled'
    # cb2['state']='disabled'
    b1['state'] = 'disabled'
    b2['state'] = 'disabled'


def genFiles(root, v, rb1, rb2, b3):
    global patient_type
    rb1['state'] = 'disabled'
    rb2['state'] = 'disabled'
    b3['state'] = 'disabled'

    patient_type = v.get()

    set_global_vals(v)
    vals = []

    maxVal = 83 if patient_type == 'C' else 114

    vals = [i + 1 for i in range(maxVal)]

    # b1 = tk.Button(root, text='Calculate (Default Weights)', command=lambda: calc_mod(root, i=0, type=patient_type))
    # b1.grid(row=5, column=0, sticky='W')
    b2 = tk.Button(root, text='Change Weights', command=lambda: modWeights(root, v, b2, b_load))
    b2.grid(row=5, column=0, sticky='W')
    b_load = tk.Button(root, text='Load Weights', command=lambda: loading_weights(root))
    b_load.grid(row=5, column=1, sticky='W', padx=(10, 0))


def loading_weights(root):
    k = tk.filedialog.askopenfilename(initialdir=base_root + "SmartHealthVisualization\\Saved Weights",
                                      title="Select file",
                                      filetypes=(("XML Files", "*.xml"), ("all files", "*.*")))
    print(k)
    tree_loaded = et.parse(k)
    root_loaded = tree_loaded.getroot()
    for i in range(2):
        weights_loaded = []
        subweights_loaded = []
        model_loaded = root_loaded[i]
        # if model_loaded.text == 'Default':
        #     calc_mod(root, i=0)
        # else:
        for ind, feature in enumerate(model_loaded):
            toload = feature.attrib['weight']
            weights_loaded.append(float(toload))
            temp_list = []
            for sub_feature in feature:
                if sub_feature.tag != 'main_feature_weight':
                    temp_list.append(float(sub_feature.text))
            subweights_loaded.append(temp_list)
        print(weights_loaded)
        print(subweights_loaded)
        main_weights[i].append(weights_loaded)
        main_weights[i].append(subweights_loaded)
        calc_mod(root, weights=weights_loaded, loaded_subweights=subweights_loaded, i=2)


def modWeights(root, v, button1=None, button2=None):
    """
	opens after pressing (Change Weights)
	variables introduced:
		titles: all broad titles as Strings
		subtitles: all subtitles as String
		labels: a list that has all titles as check buttons
		fix: a list that contains the fix buttons
		expand: a list that contains the expand buttons
		def_val: the default weights
		checkVals: the list that contains the variables corresponding to the checkbuttons (IntVar)
	"""

    global mainSubPositions
    global subFixState

    # disableFirst(b1, b2)
    # todo: add the next two lines
    button1['state'] = 'disabled'
    button2['state'] = 'disabled'
    v_2 = str(v.get())

    titles = xp.getLabels1(v_2)
    subtitles = xp.getLabels2(v_2)
    subtitles[0].pop(0)

    for i in range(len(subWeights)):
        subWeights[i] = [1 / len(subtitles[i]) for a in range(len(subtitles[i]))]
        subFixState[i] = [0 for a in range(len(subtitles[i]))]

    CumSum = 0
    for i in range(len(mainSubPositions)):
        mainSubPositions[i] = [0 for k in range(len(subtitles[i]))]
        for j in range(len(subtitles[i])):
            # mainSubPositions[i][j]=mainPositions[i]+j+1 if i==0 else mainSubPositions[i-1][-1]+2+j
            mainSubPositions[i][j] = mainPositions[i] + j + 1
    # print(mainSubPositions)

    # todo: add the following lines
    global tracking_label
    tracking_label = tk.Label(master=root, text="Choose 1st Feature(s):", font='Helvetica 10 bold')
    tracking_label.grid(row=6, column=0, sticky='W')

    m = len(titles)

    r = 7  # the row number that we reached with the other pieces of code
    var = [0 for i in range(m)]  # for expanding
    labels = []
    weights = []
    fix = []
    expand = []
    # if v_2 == 'C':
    def_val = str(1 / (len(titles)))
    checkVals = []
    for i in range(len(titles)):
        checkVals.append(tk.IntVar(root, value=1))

    for ind, title in enumerate(titles):
        rowNum = r + ind
        # Labels
        labels.append(tk.Checkbutton(root, text=title, var=checkVals[ind],
                                     command=lambda a=ind: checked(a, fix, weights, labels, checkVals)))
        labels[ind].grid(row=rowNum, sticky='W')
        # Weights entry
        weights.append(tk.Entry(root))
        weights[ind].insert(0, def_val)
        weights[ind].grid(row=rowNum, column=2, sticky='EW')
        # Fix button
        fix.append(tk.Button(root, text=' Fix ', command=lambda nn=ind: updateAfterFix(nn, weights, fix)))
        fix[ind].grid(row=rowNum, column=3, sticky='W')
        # expand button
        expand.append(tk.Button(root, text='Expand',
                                command=lambda z=ind: expandList(root, z, subtitles, var, labels, weights, fix,
                                                                 expand)))
        expand[ind].grid(row=rowNum, column=1, sticky='W')

    print('Main Weights: ', [w.get() for w in weights])

    # alpha = tk.StringVar(root)
    #
    # alpha_value = tk.Scale(root, from_=0, to=1, orient="horizontal", length=400, variable=alpha,
    #                        label='The Weight of Similarity (vs. Granularity) - α', resolution=0.01,
    #                        tickinterval=0.25)
    # alpha_value.grid(row=62, column=0, columnspan=2, sticky='W')
    # alpha_value.set(0.8)

    refButton = tk.Button(root, text='Refresh Weights', command=lambda: refreshButton(weights, fix))
    refButton.grid(row=60, column=2)
    global confirm_button
    confirm_button = tk.Button(root, text='Confirm 1st criteria', command=lambda: calc_mod(root, weights,fix=fix))
    confirm_button.grid(row=60, sticky='W')


def checked(ind, fix, weights, labels, checkVals):
    '''
	The method is called when a broad category is checked/unchecked
	'''
    if checkVals[ind].get() == 0:
        # If we uncheck:
        if fixState[ind] == 0:
            # Zero the weight
            weights[ind].delete(0, len(weights[ind].get()))
            weights[ind].insert(0, '0')
            # Fix this zero
            updateAfterFix(ind, weights, fix)
        # In case its value was fixed, unfix it
        elif fixState[ind] == 1:
            # if it was fixed --> unfix it
            updateAfterFix(ind, weights, fix)
            weights[ind].delete(0, len(weights[ind].get()))
            weights[ind].insert(0, '0')
            # fix this zero
            updateAfterFix(ind, weights, fix)
        fix[ind]['state'] = 'disabled'
    elif checkVals[ind].get() == 1:
        # If we check:
        updateAfterFix(ind, weights, fix)
        fix[ind]['state'] = 'normal'


def updateAfterFix(ind, weights, fix):
    '''
	called upon clicking the fix/unfix button OR after checking/unchecking a broad category
	'''
    if fixState[ind] == 0:
        # If we are fixing:
        fix[ind]['text'] = 'Unfix'
        weights[ind]['state'] = 'disabled'
        fixedValues = 0
        countFree = 0
        fixState[ind] = 1

        for i in range(len(weights)):
            if fixState[i] == 1:
                fixedValues += float(weights[i].get())
            else:
                countFree += 1
        remValue = (1 - fixedValues) / countFree  # to adjust the rest of the unfixed weights
        for i in range(len(weights)):
            if fixState[i] == 0:
                weights[i].delete(0, len(weights[i].get()))
                weights[i].insert(0, remValue)
    else:
        # if we are unfixing:
        fix[ind]['text'] = ' Fix '
        weights[ind]['state'] = 'normal'
        fixState[ind] = 0


def refreshButton(weights, fix):
    a = 0
    sum = 0
    # for f in fixState:
    #   if f == 0:
    #       a +=
    for i in range(len(fixState)):
        if fixState[i]==1:
            updateAfterFix(i, weights,fix)
        subRef(i)
    # for ind, weight in enumerate(weights):
    #     if fixState[ind] == 1:
    #         sum += float(weight.get())

    # rem = (1 - sum) / a
    for ind, weight in enumerate(weights):
        weight.delete(0, len(weight.get()))
        weight.insert(0, 1/len(weights))


def expandList(root, ind, subtitles, var, labels, weights, fix, expand):
    '''
	the method is called when the expand/collapse button is pressed
	'''
    resize(root)
    global saveButton
    global subRefreshButton
    global newWindow
    global subCheckList

    error = False
    # if we expand
    # we first check if another title is expanded:
    for i in range(len(expState)):
        if i != range and expState[i] == 1:
            error = True
    if error:
        messagebox.showerror("Invalid expansion", "Close the other expansions first")

    else:
        '''
		Construction Area Starts
		'''

        newWindow = tk.Tk()
        newWindow.title('SubWeights')
        newWindow.protocol("WM_DELETE_WINDOW", lambda: subQuit(ind, expand))
        newWindow.tk.call('tk', 'scaling', 2.0)

        headLabel = tk.Label(newWindow, text=str(labels[ind].cget("text")))
        headLabel.grid(row=0, column=0, sticky='W')

        '''
		Construction Area Ends
		'''
        curr_subs = subtitles[ind]  # get the corresponding subtitles as strings
        # currentRow = labels[ind].grid_info().get('row') #get the row of the expanded category
        # currentRow = mainPositions[ind]+var[ind]

        '''


		for i in range(ind+1, len(var)):
			#************************************************************************************
			#* editing the var list at the corresponding index to add the length of the current *
			#* list of subtitles to all the indeces after the expanded one (the affected ones)  *
			#************************************************************************************
			var[i] += len(curr_subs) if len(curr_subs)<6 else 6
		'''
        subFixState[ind] = [0 for i in range(len(curr_subs))]
        #	A variable to hold the currently-added checkbuttons (to be appended later to the global variable)
        subTicks = []
        count = 0
        for k, sub in enumerate(curr_subs):
            count = int(k / 6)
            remCount = int(k % 6)

            subCheckList[ind].insert(k, tk.IntVar(newWindow, value=1))

            subTicks.insert(k, tk.Checkbutton(newWindow, text=sub, var=subCheckList[ind][k],
                                              command=lambda a=k: subChecked(ind, a)))
            #	--		subTicks[k].grid(row = mainSubPositions[ind][k%6]+var[ind], column=1+(3*count), sticky='W')
            subTicks[k].grid(row=1 + remCount, column=(3 * count), sticky='W')

            subWeightsBox[ind].insert(k, tk.Entry(newWindow))
            subWeightsBox[ind][k].insert(0, str(subWeights[ind][k]))
            #	--		subWeightsBox[ind][k].grid(row= mainSubPositions[ind][k%6]+var[ind], column=2+(3*count), sticky='EW')
            subWeightsBox[ind][k].grid(row=1 + remCount, column=1 + (3 * count), sticky='EW')

            subFixButtons[ind].insert(k, tk.Button(newWindow, text=' Fix ', command=lambda m=k: subFix(ind, m)))
            #	--		subFixButtons[ind][k].grid(row= mainSubPositions[ind][k%6]+var[ind], column=3+(3*count), sticky='W')
            subFixButtons[ind][k].grid(row=1 + remCount, column=2 + (3 * count), sticky='W')

        #	Insert the list of checkbuttons to the global list
        subs.pop(ind)
        subs.insert(ind, subTicks)
        # putInPlace(labels, weights, fix, expand, subs, var, ind)
        expState[ind] = 1
        expand[ind]['text'] = 'Expanded'
        expand[ind]['state'] = 'disabled'
        # print('var list has become: {0} in expandList method (line 254)'.format(var))

        saveButton = tk.Button(newWindow, text='Save and Quit ', command=lambda: saveVals(ind, expand))
        #	--saveButton.grid(row= labels[ind].grid_info().get('row')+1, column=0, sticky='W')
        saveButton.grid(row=7, column=0, sticky='W')

        subRefreshButton = tk.Button(newWindow, text='Refresh SubWeights ', command=lambda: subRef(ind))
        #	--subRefreshButton.grid(row=labels[ind].grid_info().get('row')+2, column=0, sticky='W')
        subRefreshButton.grid(row=7, column=1, sticky='W')

        subResetButton = tk.Button(newWindow, text='Reset SubWeights ', command=lambda: subReset(ind))
        subResetButton.grid(row=7, column=2, sticky='W')

        quitButton = tk.Button(newWindow, text='Quit without Saving', command=lambda: subQuit(ind, expand))
        quitButton.grid(row=7, column=3, sticky='W')


def subFix(ind, m):
    if subFixState[ind][m] == 0:
        # If we are fixing:
        subFixButtons[ind][m]['text'] = 'Unfix'
        subWeightsBox[ind][m]['state'] = 'disabled'
        fixedValues = 0
        countFree = 0
        subFixState[ind][m] = 1

        for i in range(len(subWeightsBox[ind])):
            if subFixState[ind][i] == 1:
                fixedValues += float(subWeightsBox[ind][i].get())
            else:
                countFree += 1
        remValue = (1 - fixedValues) / countFree  # to adjust the rest of the unfixed weights
        for i in range(len(subWeights[ind])):
            if subFixState[ind][i] == 0:
                subWeightsBox[ind][i].delete(0, len(subWeightsBox[ind][i].get()))
                subWeightsBox[ind][i].insert(0, remValue)
    else:
        # if we are unfixing:
        subFixButtons[ind][m]['text'] = ' Fix '
        subWeightsBox[ind][m]['state'] = 'normal'
        subFixState[ind][m] = 0


def saveVals(ind, expand):
    sum = 0
    # print(' in the save method this is subWeightsBox[ind]: {0}'.format(subWeightsBox[ind]))
    for i in range(len(subWeightsBox[ind])):
        # print('i ', i)
        # print('num: ', float(subWeightsBox[ind][i].get()))
        sum += float(subWeightsBox[ind][i].get())
        # print('done ', i)
    if not (round(sum, 3) > 0.99 and round(sum, 3) <= 1):
        # print('error')
        messagebox.showerror("Invalid Values", "Values inserted don't add up to 1")
    else:
        for i in range(len(subWeightsBox[ind])):
            subWeights[ind][i] = float(subWeightsBox[ind][i].get())
    print(subWeights)
    subQuit(ind, expand)


def subQuit(ind, expand):
    # print('quit')
    subCheckList[ind] = []
    subWeightsBox[ind] = []
    subFixButtons[ind] = []

    subFixState[ind] = []
    newWindow.destroy()
    expState[ind] = 0
    expand[ind]['state'] = 'normal'
    expand[ind]['text'] = 'Expand'


def subReset(ind):
    global subs
    if subs[ind] is not None:
        if len(subs[ind])!=0:
            for i in range(len(subs[ind])):
                if int(subCheckList[ind][i].get()) == 0:
                    subs[ind][i].select()
                    subChecked(ind, i)
                elif subFixState[ind][i] == 1:
                    subFix(ind, i)
            subRef(ind)
        else:
            pass



def subRef(ind):
    fixedValues = 0
    countFree = 0
    if len(subWeightsBox[ind])==0:
        for i in range(len(subWeights)):
            subWeights[i] = [1 / len(subWeights[i]) for a in range(len(subWeights[i]))]
    else:
        for i in range(len(subWeightsBox[ind])):
            if subFixState[ind][i] == 1:
                fixedValues += float(subWeightsBox[ind][i].get())
                # print('at ind={0} we entered the else and fixedValues became {1}'.format(ind, fixedValues))

            else:
                countFree += 1
                # print('at ind={0} we entered the else and countfree became {1}'.format(ind, countFree))
        remValue = (1 - fixedValues) / countFree  # to adjust the rest of the unfixed weights
        for i in range(len(subWeightsBox[ind])):
            subWeightsBox[ind][i].delete(0, len(subWeightsBox[ind][i].get()))
            subWeightsBox[ind][i].insert(0, remValue)


def subChecked(ind, m):
    if subCheckList[ind][m].get() == 0:
        # If we uncheck:
        if subFixState[ind][m] == 0:
            # Zero the weight
            subWeightsBox[ind][m].delete(0, len(subWeightsBox[ind][m].get()))
            subWeightsBox[ind][m].insert(0, '0')
            # Fix this zero
            subFix(ind, m)
        # In case its value was fixed, unfix it
        elif subFixState[ind][m] == 1:
            # if it was fixed --> unfix it
            subFix(ind, m)
            subWeightsBox[ind][m].delete(0, len(subWeightsBox[ind][m].get()))
            subWeightsBox[ind][m].insert(0, '0')
            # fix this zero
            subFix(ind, m)
        subFixButtons[ind][m]['state'] = 'disabled'
    elif subCheckList[ind][m].get() == 1:
        # If we check:
        subFix(ind, m)
        subFixButtons[ind][m]['state'] = 'normal'


def default_weight(patient_type, ind):
    if ind == 1:
        length = len(xp.getLabels1(patient_type))
        toreturn = []
        for a in range(length):
            toreturn.append(1 / length)
        return toreturn
    elif ind == 2:
        list_of_subs = xp.getLabels2(patient_type)
        list_of_subs[0].pop(0)
        toreturn = []
        for sub in list_of_subs:
            length = len(sub)
            temp_list = []
            for a in range(length):
                temp_list.append(1 / length)
            toreturn.append(temp_list)
        return toreturn


def calc_mod(root, weights=None, i=1, loaded_subweights=None, type=None, fix=None):
    global dend_index
    global sim_matrices
    global NB_ELEMENTS
    global main_weights
    global subWeights
    if dend_index < 2:
        fl_weights = []  # array of floating weights
        if i == 1:  # modified weights
            sum = 0
            for weight in weights:
                fl_weights.append(float(weight.get()))
                sum += float(weight.get())
            if not (0.99 < round(sum, 3) <= 1):
                # print('error')
                messagebox.showerror("Invalid Values", "Values inserted don't add up to 1")
                refreshButton(weights, fix)
                return
        if i == 2:  # loaded
            fl_weights = weights
            subWeights = loaded_subweights

        if len(fl_weights) != 0:
            main_weights[dend_index].append(fl_weights)
            main_weights[dend_index].append(subWeights)
        else:
            main_weights[dend_index].append(default_weight(patient_type, 1))
            main_weights[dend_index].append(default_weight(patient_type, 2))
            # print('Default added at index ', dend_index)
        v_2 = str(v.get())
        tic = time.perf_counter()
        simMatrix, n = xp.create_patient_distance_matrix(v_2, fl_weights, subWeights) if i != 0 else \
            xp.create_patient_distance_matrix(v_2)
        toc = time.perf_counter()
        NB_ELEMENTS = n
        t = tk.Label(root, text="")
        t.grid(row=65, column=0, sticky='W')
        # xp.cluster(simMatrix)
        sim_matrices.append(simMatrix)
        dend_index += 1
        if dend_index == 2:
            # Got sim matrices
            dend_index += 1
            root.destroy()
            popup_alpha_window()
        else:
            tracking_label['text'] = 'Choose 2nd Feature(s):'
            confirm_button['text'] = 'Confirm & Proceed To Visualization'
            refreshButton(weights,fix)


def popup_alpha_window(windows=None):
    global alpha_g
    alpha_window = tk.Tk()
    alpha_variable = tk.DoubleVar(alpha_window, value=alpha_g)
    s = tk.Scale(alpha_window, label='Choose an α- value:', from_=0, to=1,
                 orient=tk.HORIZONTAL, length=300, showvalue=0.8,
                 tickinterval=0.5, var=alpha_variable,
                 resolution=0.01)
    s.grid(row=0, column=0)
    if windows is None:
        windows = [alpha_window]
    else:
        windows.append(alpha_window)
    b = tk.Button(alpha_window, text='Confirm', command=lambda: plot_window(alpha_variable.get(), windows=windows))
    b.grid(row=1, column=0)


def resize(root):
    for r in range(13):
        root.columnconfigure(r, weight=3)


''' Section below deals with the plotting process '''


def plot_window(alpha, windows=None):
    if windows is not None:
        for window in windows:
            window.destroy()
    global sim_matrices
    global threshold_value
    global min_nodes_threshold
    global alpha_g

    alpha_g = alpha

    # This defines the Python GUI backend to use for matplotlib
    matplotlib.use('TkAgg')

    graph_kw = {'slider1': NB_ELEMENTS, 'slider2': NB_ELEMENTS, 'annotation': 1, 'link': 1, 'patch_color': 'Red',
                'threshold': 0, 'nb_nodes_threshold': None}
    # Initialize an instance of Tk
    plots_window = tk.Tk()
    plots_window.tk.call('tk', 'scaling', 2.0)

    plots_window.title('Dendrograms')

    # Initialize matplotlib figure for graphing purposes
    fig = plt.figure(figsize=(12, 8))

    # Special type of "canvas" to allow for matplotlib graphing
    canvas = FigureCanvasTkAgg(fig, master=plots_window)
    plot_widget = canvas.get_tk_widget()
    plot_widget.grid(row=0, column=0, columnspan=6, sticky='EW')
    # plots_window.grid_columnconfigure(0, weight=4)
    toolbarFrame = tk.Frame(master=plots_window)
    toolbarFrame.grid(row=1, column=1, columnspan=2, sticky='EW')
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    # print('sim matrices: ', sim_matrices)
    mv = xp.get_model_variants(sim_matrices[0], sim_matrices[1])

    optimal_visualization = OptimalVisualization(mv, alpha, NB_ELEMENTS)
    zoom1, zoom2 = optimal_visualization.calculate_scores()  # commented for experiments
    # zoom1 = 7
    # zoom2 = 6
    # print('zoom1: {0}, zoom2: {1}'.format(zoom1, zoom2))
    zoom_vals.append(zoom1)
    zoom_vals.append(zoom2)

    # Add the plot to the tkinter widget
    plot_widget.grid(row=0, column=0, columnspan=3)

    apply = tk.Button(plots_window, text='Set Preferences',
                      command=lambda: preferences_popup())

    apply.grid(row=2, column=1)

    # menus
    menubar = tk.Menu(plots_window)
    plots_window.config(menu=menubar)

    # edit menu creation
    optionsmenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Options', menu=optionsmenu)

    optionsmenu.add_command(label='Export Weights', command=lambda: export_weights(1))

    viewmenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='View', menu=viewmenu)

    viewmenu.add_command(label='View Weights', command=lambda: export_weights(2))

    plot_dendrograms(mv, zoom_percentage1=zoom1, zoom_percentage2=zoom2)

    for r in range(8):
        plots_window.columnconfigure(r, weight=3)

    def preferences_popup():
        def update_kwargs():
            global zoom_vals
            graph_kw['slider1'] = slider1.get()
            graph_kw['slider2'] = slider2.get()
            graph_kw['annotation'] = annotation_status.get()
            graph_kw['link'] = connection_type.get()
            graph_kw['patch_color'] = patch_color.get()
            graph_kw['threshold'] = threshold_value
            graph_kw['nb_nodes_threshold'] = min_nodes_threshold
            zoom_vals[0] = slider1.get()
            zoom_vals[1] = slider2.get()
            update_graph(mv, fig, **graph_kw)

        preferences_window = tk.Tk()
        preferences_window.tk.call('tk', 'scaling', 2.0)
        preferences_window.title('Choose the graph preferences')

        tk.Label(preferences_window, text='Zooming levels: ').grid(row=0, column=0)
        slider1 = tk.Scale(preferences_window, from_=1, to=NB_ELEMENTS, orient="horizontal", label='Nodes Shown 1',
                           length=150, command=lambda x=None: update_kwargs())
        slider1.grid(row=1, column=0)
        slider1.set(zoom_vals[0])

        slider2 = tk.Scale(preferences_window, from_=1, to=NB_ELEMENTS, orient="horizontal", label='Nodes Shown 2',
                           length=150, command=lambda x=None: update_kwargs())
        slider2.grid(row=1, column=3)
        slider2.set(zoom_vals[1])

        connection_type = tk.IntVar(preferences_window, value=graph_kw['link'])

        tk.Label(preferences_window, text='Connection Type: ').grid(row=2, column=0)
        off_connection_type = tk.Radiobutton(preferences_window, text="Off", variable=connection_type, value=0,
                                             command=lambda: update_kwargs())
        default_connection_type = tk.Radiobutton(preferences_window, text="Default", variable=connection_type, value=1,
                                                 command=lambda: update_kwargs())
        set_threshold_connection_type = tk.Radiobutton(preferences_window, text="Threshold", variable=connection_type,
                                                       value=2,
                                                       command=lambda: popup_threshold_window(mv, fig,
                                                                                              slider1=slider1.get(),
                                                                                              slider2=slider2.get(),
                                                                                              annotation=annotation_status.get(),
                                                                                              link=connection_type.get(),
                                                                                              patch_color=patch_color.get()))

        off_connection_type.grid(row=3, column=0)
        default_connection_type.grid(row=3, column=1)
        set_threshold_connection_type.grid(row=3, column=2)

        tk.Button(preferences_window, text='Minimum Number of Nodes to Link',
                  command=lambda: popup_threshold_window(mv, fig, slider1=slider1.get(),
                                                         slider2=slider2.get(),
                                                         annotation=annotation_status.get(),
                                                         link=connection_type.get(),
                                                         patch_color=patch_color.get(),
                                                         type='min nodes threshold')).grid(row=9, column=0,
                                                                                           columnspan=3)

        window_list = [preferences_window, plots_window]
        tk.Button(preferences_window, text='Plot Default Zooming',
                  command=lambda: popup_alpha_window(windows=window_list)
                  ).grid(row=9, column=3)

        patch_color = tk.StringVar(preferences_window, value=graph_kw['patch_color'])
        tk.Label(preferences_window, text='Connection Color: ').grid(row=4, column=0)

        red_radiobutton = tk.Radiobutton(preferences_window, text='Red', variable=patch_color, value='Red',
                                         command=lambda: update_kwargs())
        green_radiobutton = tk.Radiobutton(preferences_window, text='Green', variable=patch_color, value='Green',
                                           command=lambda: update_kwargs())
        blue_radiobutton = tk.Radiobutton(preferences_window, text='Blue', variable=patch_color, value='Blue',
                                          command=lambda: update_kwargs())
        cyan_radiobutton = tk.Radiobutton(preferences_window, text='Cyan', variable=patch_color, value='Cyan',
                                          command=lambda: update_kwargs())
        magenta_radiobutton = tk.Radiobutton(preferences_window, text='Magenta', variable=patch_color, value='Magenta',
                                             command=lambda: update_kwargs())
        yellow_radiobutton = tk.Radiobutton(preferences_window, text='Yellow', variable=patch_color, value='Yellow',
                                            command=lambda: update_kwargs())
        black_radiobutton = tk.Radiobutton(preferences_window, text='Black', variable=patch_color, value='Black',
                                           command=lambda: update_kwargs())

        red_radiobutton.grid(row=5, column=0)
        green_radiobutton.grid(row=5, column=1)
        blue_radiobutton.grid(row=5, column=2)
        cyan_radiobutton.grid(row=5, column=3)
        magenta_radiobutton.grid(row=6, column=0)
        yellow_radiobutton.grid(row=6, column=1)
        black_radiobutton.grid(row=6, column=2)

        annotation_status = tk.IntVar(preferences_window, value=graph_kw['annotation'])

        tk.Label(preferences_window, text='Annotation Type: ').grid(row=7, column=0)

        annotation_on_radio = tk.Radiobutton(preferences_window, text='On', variable=annotation_status, value=1,
                                             command=lambda: update_kwargs())
        annotation_left_radio = tk.Radiobutton(preferences_window, text='Left', variable=annotation_status, value=2,
                                               command=lambda: update_kwargs())
        annotation_right_radio = tk.Radiobutton(preferences_window, text='Right', variable=annotation_status, value=3,
                                                command=lambda: update_kwargs())
        annotation_off_radio = tk.Radiobutton(preferences_window, text='Off', variable=annotation_status, value=4,
                                              command=lambda: update_kwargs())

        annotation_on_radio.grid(row=8, column=0)
        annotation_left_radio.grid(row=8, column=1)
        annotation_right_radio.grid(row=8, column=2)
        annotation_off_radio.grid(row=8, column=3)

        preferences_window.mainloop()

    plots_window.grid_rowconfigure(0, weight=2)
    plots_window.grid_columnconfigure(0, weight=2)
    plots_window.grid_columnconfigure(1, weight=2)

    plots_window.mainloop()


def export_weights(type=1):
    global main_weights
    f = None
    if type == 1:
        my_path = base_root+ "SmartHealthVisualization\\SavedWeights"

        '''
        file_numbers = [f.split("weights_", 1)[1][:-4] for f in os.listdir(my_path) if
                        os.path.isfile(os.path.join(my_path, f))]
        file_numbers.sort()
        index = int(file_numbers[-1])
        if len(file_numbers) == 0:
            index = 0
        '''

        f = open(my_path + "\weights_" +
                 str(datetime.now().date()).replace('-', '') + '_' + str(datetime.now().hour) + str(
            datetime.now().minute)
                 + ".xml", "w")

    letter = patient_type
    # print('letter is: ', letter)
    x = main_weights
    main_titles_list = xp.getLabels1(letter, True)
    # print(main_titles_list)
    sub_titles_lists = xp.getLabels2(letter, True)
    root_node = et.Element('root')
    model_nodes = [et.SubElement(root_node, 'model', {'index': str(i)}) for i in range(2)]
    for j in range(2):
        # if main_weights[j][0] != 'Default':
        main_nodes = [et.SubElement(model_nodes[j], main_title, {'weight': str(main_weights[j][0][ind])}) for
                      ind, main_title in enumerate(main_titles_list)]
        for ind, main_node in enumerate(main_nodes):
            sub_nodes = [et.SubElement(main_node, sub_titles_lists[ind][k]) for k in
                         range(len(sub_titles_lists[ind]))]

            for k in range(len(main_weights[j][1][ind])):
                sub_nodes[k].text = str(main_weights[j][1][ind][k])

        # else:
        #     model_nodes[j].text = 'Default'

    # print('root node [0] ', str(prettify(root_node[0])))
    if (type == 1):
        f.write(str(prettify(root_node)))
        f.close()
    if (type == 2):
        weights_window = tk.Tk()

        weights_window.grid_rowconfigure(0, weight=2)
        weights_window.columnconfigure(0, weight=3)
        weights_window.title('Show Weights')

        # Create a frame for the canvas with non-zero row&column weights
        frame_canvas = tk.Frame(weights_window)
        frame_canvas.grid(row=0, column=0, pady=(5, 0), sticky='nw')
        frame_canvas.grid_rowconfigure(0, weight=1)
        frame_canvas.grid_columnconfigure(0, weight=1)

        # Add a canvas in that frame
        canvas = tk.Canvas(frame_canvas)
        canvas.grid(row=0, column=0, sticky="news")

        frame_labels = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame_labels, anchor='nw')

        label1 = tk.Label(frame_labels, text=str(prettify(root_node[0])))
        label1.grid(row=1, column=1, pady=(5, 0), sticky='nw')

        label2 = tk.Label(frame_labels, text=str(prettify(root_node[1])))
        label2.grid(row=1, column=2)

        frame_labels.update_idletasks()

        # Link a scrollbar to the canvas
        vsb = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
        vsb.grid(row=0, column=1, sticky='ns')
        canvas.configure(yscrollcommand=vsb.set)

        hsb = tk.Scrollbar(frame_canvas, orient='horizontal', command=canvas.xview)
        hsb.grid(row=2, column=0, sticky='ew')
        canvas.configure(xscrollcommand=hsb.set)

        # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
        frame_canvas.config(width=label1.winfo_width() + label2.winfo_width() + vsb.winfo_width(),
                            height=label1.winfo_height())

        # Set the canvas scrolling region
        canvas.config(scrollregion=canvas.bbox("all"))


def prettify(elem):
    """
    :param: elem: the element tree elem
    :return: a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def popup_threshold_window(mv, fig, slider1, slider2, annotation, link, patch_color, type='threshold value'):
    global threshold_value, min_nodes_threshold, hovering_on, percentage_nodes_threshold
    popup_threshold = tk.Tk()
    popup_threshold.tk.call('tk', 'scaling', 2.0)
    popup_threshold.title(
        'Choose a Threshold Value' if type == 'threshold' else 'Minimum number of nodes to display link')
    if type == 'threshold value':
        temp_variable = tk.DoubleVar(popup_threshold)
        slider = tk.Scale(popup_threshold, from_=0, to=1, resolution=0.01, length=500, var=temp_variable,
                          orient="horizontal",
                          label='Choose a value for the minimal connection strength')
        slider.set(threshold_value)
    else:
        temp_variable = tk.IntVar(popup_threshold)
        slider = tk.Scale(popup_threshold, from_=0, to=NB_ELEMENTS, resolution=1, length=500, var=temp_variable,
                          orient='horizontal', label='Choose minimum number of elements in node to display link',
                          )
        slider.set(min_nodes_threshold)
        tick_box_variable = tk.IntVar(popup_threshold, hovering_on)
        tick_box = tk.Checkbutton(popup_threshold, text='Display links when hovering over centers',
                                  variable=tick_box_variable)
        tick_box.grid(row=1, column=0, columnspan=9)
        # percentage_temp_variable = tk.IntVar(popup_threshold)
        # slider_percentage = tk.Scale(popup_threshold, from_=0, to=100, resolution=1, length=500,
        #                              var=percentage_temp_variable,
        #                              orient='horizontal', label='Choose percentage of nodes to link',
        #                              command=lambda x=None: fix_value('number'))
        # slider_percentage.set(percentage_nodes_threshold)
        # slider_percentage.grid(row=2, column=0, columnspan=9)

    slider.grid(row=0, column=0, columnspan=9)
    b = tk.Button(popup_threshold, text='Confirm',
                  command=lambda: button_command(mv, fig, slider1, slider2, annotation, link, patch_color))
    b.grid(row=3, column=4)

    # def fix_value(type):
    # if type == 'number':
    # temp_variable.set(int(percentage_temp_variable.get() * nb_links / 100))
    # print('temp variable', temp_variable.get())
    # print('percentage temp variable', percentage_temp_variable.get())
    # if type == 'percentage':
    # percentage_temp_variable.set(int(temp_variable.get() * 100 / nb_links))
    # print('percentage temp variable', percentage_temp_variable.get())
    # print('temp vairable', temp_variable.get())

    print('nb links: ', nb_links)

    def button_command(*args):
        global threshold_value, min_nodes_threshold, hovering_on
        popup_threshold.destroy()
        if type == 'threshold value':
            threshold_value = temp_variable.get()
        else:
            min_nodes_threshold = temp_variable.get()
            hovering_on = tick_box_variable.get()
        update_graph(*args, threshold=threshold_value, nb_nodes_threshold=min_nodes_threshold)


def update_graph(mv, fig, slider1=NB_ELEMENTS, slider2=NB_ELEMENTS, annotation=1, link=1, patch_color='Red',
                 threshold=0, nb_nodes_threshold=None):
    if nb_nodes_threshold is None:
        nb_nodes_threshold = min_nodes_threshold
    plt.clf()
    plot_dendrograms(mv, annotation=annotation, zoom_percentage1=slider1, zoom_percentage2=slider2, fig=fig,
                     link_on=link,
                     patch_color=patch_color, threshold=threshold, nb_nodes_threshold=nb_nodes_threshold)


def plot_dendrograms(models, annotation=1, zoom_percentage1=None, zoom_percentage2=None, fig=None, link_on=1,
                     patch_color='Red', threshold=0, nb_nodes_threshold=1):
    global zoom_vals, NB_ELEMENTS, threshold_value, nb_links

    if zoom_percentage1 is None:
        zoom_percentage1 = NB_ELEMENTS
    if zoom_percentage2 is None:
        zoom_percentage2 = NB_ELEMENTS

    # print('zoom_percentage1: {0}, zoom_percentage2: {1}'.format(zoom_percentage1, zoom_percentage2))

    plt.clf()
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.set_title('Dendrogram #1: ')
    ax2.set_title('Dendrogram #2: ')

    p1 = zoom_percentage1
    p2 = zoom_percentage2

    zoom_vals[0] = zoom_percentage1
    zoom_vals[1] = zoom_percentage2

    set_link_color_palette(['silver'])
    # print('Showing Dendrograms: ')
    # print('first dendrogram at zooming {},{}:'.format(p1, p2))
    linkage1 = xp.fix_linkage_matrix(xp.getLinkageMatrix(models[0]))
    d1 = dendrogram(linkage1, orientation='left',
                    p=p1, truncate_mode='lastp', ax=ax1, color_threshold=2.0)

    # print('second dendrogram at zooming {},{}:'.format(p1, p2))

    linkage2 = xp.fix_linkage_matrix(xp.getLinkageMatrix(models[1]))
    d2 = dendrogram(linkage2, orientation='right',
                    p=p2, truncate_mode='lastp', ax=ax2, color_threshold=2.0)

    current_figure = plt.gcf()

    current_figure.tight_layout(pad=6.0)
    # print('model children', models[0].children_)
    # print('leaves #1: ', d1)

    list_of_centers = [DendrogramTools.get_dend_centers(d1), DendrogramTools.get_dend_centers(d2)]

    basic_components = [DendrogramTools.get_basic_components(models[0], NB_ELEMENTS),
                        DendrogramTools.get_basic_components(models[1], NB_ELEMENTS)]

    # What time is it? it's annotation time!
    offset1 = NB_ELEMENTS - p1
    offset2 = NB_ELEMENTS - p2

    if link_on != 0:
        # print('List of centers', list_of_centers)
        # print('basic components', basic_components)

        # for connection_patches: 0 is for dynamic, 1 is for static

        connection_patches, circles_drawn, avg = LinksGeneration.connect_components(basic_components, list_of_centers,
                                                                                    ax1, ax2, patch_color,
                                                                                    status='exact' if link_on == 1 else 'different',
                                                                                    threshold=0.5 if link_on == 1 else threshold,
                                                                                    number_of_elements_hide_link=nb_nodes_threshold
                                                                                    )

        nb_links = len(connection_patches[0]) + len(connection_patches[1])

        if threshold_value == 0:  # first time
            threshold_value = 0.5

        artists_index_tracker = 0

        for i in range(len(connection_patches[0])):
            current_figure.add_artist(connection_patches[0][i][0])
            current_figure.artists[artists_index_tracker].set_visible(False)
            artists_index_tracker += 1

        dynamic_artists_index_stop = artists_index_tracker

        for i in range(len(connection_patches[1])):
            current_figure.add_artist(connection_patches[1][i][0])
            artists_index_tracker += 1

        for circle_drawn in circles_drawn[0]:
            ax1.add_artist(circle_drawn[0])
        for circle_drawn in circles_drawn[1]:
            ax2.add_artist(circle_drawn[0])

        count = [0 for a in range(len(connection_patches[0]))]

        current_figure.canvas.mpl_connect("motion_notify_event", lambda event: hover(event, circles_drawn,
                                                                                     connection_patches,
                                                                                     current_figure, count,
                                                                                     dynamic_artists_index_stop,
                                                                                     [zoom_percentage1,
                                                                                      zoom_percentage2],
                                                                                     [ax1, ax2]))

    if annotation == 1 or annotation == 2:
        annotate1_on = 1
    else:
        annotate1_on = 0

    if annotation == 1 or annotation == 3:
        annotate2_on = 1
    else:
        annotate2_on = 0

    if annotate1_on == 1:
        # print('Basic Components: ', basic_components)
        for i, center in enumerate(list_of_centers[0]):
            ax1.annotate(split_array(basic_components[0][i + offset1]), list_of_centers[0][i], annotation_clip=False,
                         bbox=dict(boxstyle="round", fc="w"))

    if annotate2_on == 1:
        for i, center in enumerate(list_of_centers[1]):
            ax2.annotate(split_array(basic_components[1][i + offset2]), list_of_centers[1][i], annotation_clip=False,
                         bbox=dict(boxstyle="round", fc="w"))
    #   fig.subplots_adjust(right=1.5, left=1)

    if fig is not None:
        fig.canvas.draw()


def split_array(text):
    new_text = ''
    for i in range(0, len(text), 10):
        if i != 0:
            new_text = new_text + "\n"
        new_text += str(text[i:min(i + 10, len(text))])

    return new_text


def hover(event, circles_lists, connection_patches, current_figure, count, dynamic_artist_index_stop, zoom_percentages,
          ax):
    """
    :param show_links_when_hovering:
    :param event: the MouseEvent variable
    :param circles_lists: lists of circles present in the figure
    :param connection_patches: the links that we have; 2d array with [dynamic links, fixed links]; [[unneccessary]]
    :param current_figure: the instance of the current figure
    :param count: an array to keep track of the number of times we are setting a link to be visible (for stability)
    :param dynamic_artist_index_stop: the artist index after which fixed links exist
    :param zoom_percentages: used to make sure we are in the proper method
    :param ax: [ax1, ax2]
    :return: void
    """
    global hovering_on
    # to rule out possibilities of old valies.
    if zoom_percentages[0] == zoom_vals[0] and zoom_percentages[1] == zoom_vals[1]:
        real_hover(event, circles_lists, connection_patches, current_figure, count, dynamic_artist_index_stop, ax,
                   show_links_when_hovering=bool(hovering_on))


def real_hover(event, circles_lists, connection_patches, current_figure, count, dynamic_artist_index_stop, ax,
               show_links_when_hovering=True):
    if event.xdata is not None:
        enlarge_factor = 1.2
        axis_index = 0
        if event.inaxes == ax[0]:
            axis_index = 1
        elif event.inaxes == ax[1]:
            axis_index = 2
        # print('event: xdata={}, ydata={} in axis: {}'.format(event.xdata, event.ydata, axis_index))
        # if axis_index == 0:
        #    raise Exception('Mouse in unknown axis')

        circles_list = circles_lists[axis_index - 1]
        for index, circle in enumerate(circles_list):

            point1 = circle[0].center[0] - (circle[0].width / 2)
            point2 = circle[0].center[0] + (circle[0].width / 2)
            point3 = circle[0].center[1] - (circle[0].height / 2)
            point4 = circle[0].center[1] + (circle[0].height / 2)

            # print('circle {} of components {}: \n{}<x<{} \n{}<y<{}'.format(axis_index, circle[1], point1, point2,
            #                                                               point3, point4))

            x_within_circle = point1 <= event.xdata <= point2
            y_within_circle = point3 <= event.ydata <= point4

            if x_within_circle and y_within_circle:
                # print('you are within circle of label: {}'.format(circle[1]))
                if show_links_when_hovering:
                    current_figure.artists[index].set_visible(True)
                if count[index] == 0:
                    for axis in ax:
                        axis.artists[index].width = axis.artists[index].width * enlarge_factor
                        axis.artists[index].height = axis.artists[index].height * enlarge_factor

                current_figure.canvas.draw_idle()
                if connection_patches[0][index][0] == current_figure.artists[index]:
                    print('****  connecting link connecting {} ****'.format(connection_patches[0][index][1]))

                count[index] += 1
                return

            elif index < len(current_figure.artists):
                if count[index] == 1:
                    if index < dynamic_artist_index_stop:
                        current_figure.artists[index].set_visible(False)
                        for axis in ax:
                            axis.artists[index].width = axis.artists[index].width / enlarge_factor
                            axis.artists[index].height = axis.artists[index].height / enlarge_factor
                        current_figure.canvas.draw_idle()
                        print('disconnecting ....')
                if count[index] > 0:
                    count[index] -= 1


def generate_dimensions(bottom_left_corner, figure_width, figure_height, figure_gap_width):
    bottom_left_corner = [bottom_left_corner[0],
                          (bottom_left_corner[0][0] + figure_width + figure_gap_width, bottom_left_corner[0][1])]

    bottom_right_corner = [(bottom_left_corner[0][0] + figure_width, bottom_left_corner[0][1]),
                           (bottom_left_corner[1][0] + figure_width, bottom_left_corner[1][1])]

    top_left_corner = [(bottom_left_corner[0][0], bottom_left_corner[0][1] + figure_height),
                       (bottom_left_corner[1][0], bottom_left_corner[1][1] + figure_height)]

    top_right_corner = [(bottom_right_corner[0][0], bottom_right_corner[0][1] + figure_height),
                        (bottom_right_corner[1][0], bottom_right_corner[1][1] + figure_height)]

    return bottom_left_corner, bottom_right_corner, top_left_corner, top_right_corner


'''Section below deals with calculating different scores to find optimal visualization'''

'''*** deleted ***'''

''' section below deals with generating links; main approach: get the jaccard similarity and act accordingly '''

'''*** deleted ***'''


def make_dpi_aware():
    print(platform.release())
    if float(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)


if __name__ == '__main__':
    welcome_screen = tk.Tk()

    w = welcome_screen.winfo_reqwidth()
    h = welcome_screen.winfo_reqheight()
    ws = welcome_screen.winfo_screenwidth()
    hs = welcome_screen.winfo_screenheight()
    x = (ws / 2) - (w / 2) - 300
    y = (hs / 2) - (h / 2) - 250
    welcome_screen.geometry('+%d+%d' % (x, y))

    spath = "images/lau.jpg"
    simg = ImageTk.PhotoImage(Image.open(spath))
    welcome_screen.minsize(height=simg.height() + 250, width=simg.width())
    welcome_screen.tk.call('tk', 'scaling', 2.0)
    welcome_screen.overrideredirect(True)
    make_dpi_aware()

    welcome_label = tk.Label(welcome_screen, text='Welcome To \nMirrored Dendrograms',
                             font=('Helvetica', 20), fg='black')
    welcome_label.grid(row=2, column=0)

    credits_label = tk.Label(welcome_screen, text='\n'
                                                  '\nA Master\'s Thesis by: Angela Moufarrej'
                                                  '\nDeveloped by: Abdulkader Fatouh'
                                                  '\nSupervised by: Dr. Joe Tekli',
                             font=('Courier', 12), fg='black')
    credits_label.grid(row=3, column=0)

    image_label = tk.Label(welcome_screen, image=simg)
    image_label.image = simg
    image_label.grid(row=0, column=0)
    v = 'None'


    def main_win():
        welcome_screen.destroy()
        root = tk.Tk()
        root.tk.call('tk', 'scaling', 2.0)
        make_dpi_aware()
        root.title('Weights Selection')
        global v
        v = tk.StringVar(root)

        v.set("C")

        tk.Label(root, text="Choose a type of patients to compare: ").grid(row=0, column=0, sticky='W')

        rb1 = tk.Radiobutton(root, text="Control", variable=v, value='C')
        rb1.grid(row=1, column=0, sticky='W')
        rb2 = tk.Radiobutton(root, text="Migraine", variable=v, value='M')
        rb2.grid(row=2, column=0, sticky='W')

        b3 = tk.Button(root, text='Confirm ', command=lambda: genFiles(root, v, rb1, rb2, b3))
        b3.grid(row=2, column=1, sticky='W')

        # restartButton = tk.Button(root, text='Restart', command=lambda: restart_program(root))
        # restartButton.grid(row=70, sticky='W')


    welcome_screen.after(5000, main_win)

    tk.mainloop()
