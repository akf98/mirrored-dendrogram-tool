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

import xml_parsing as xp

from utils import LinksGeneration
from utils import OptimalVisualization
from utils import DendrogramTools

from PIL import ImageTk, Image

#global variables
patient_type = ''

dend_index = 0
sim_matrices = []
MAX_C = 83
MAX_M = 114
subs = []
exp_state = []
fix_state = []
sub_check_list = [] #checks which sub elements are checked
main_positions = [] 
main_sub_positions = []
sub_weights_box = []
sub_weights = []
check_values = []

threshold_value = 0
min_nodes_threshold = 0
percentage_nodes_threshold = 0
iteration = 0
hovering_on = 1
tracking_label = None
confirm_button = None

zoom_vals = []

save_button = None
sub_refresh_button = None

sub_fix_buttons = []
sub_fix_state = []

new_window = None

NB_ELEMENTS = 10
nb_links = NB_ELEMENTS - 1

main_weights = [[], []]
alpha_g = 0.8

final_weights = [[[], []], [[], []]]

base_root = '\\'.join(os.getcwd().split('\\')[:-2])

def restart_program(root):
    root.destroy()
    python = sys.executable
    script = os.path.realpath(__file__)
    subprocess.Popen([python, script])

def set_global_vals(p_type):
    global exp_state, fix_state, subs, sub_check_list, main_positions, main_sub_positions, sub_weights_box, sub_weights, sub_fix_buttons, sub_fix_state

    sub_check_list = [[] for a in range(7 if str(p_type) == 'C' else 8)]
    subs = [[] for a in range(7 if str(p_type) == 'C' else 8)]
    exp_state = [0 for a in range(7 if p_type == 'C' else 8)]
    fix_state = [0 for a in range(7 if p_type == 'C' else 8)]
    main_sub_positions = [[] for a in range(7 if p_type == 'C' else 8)]
    main_positions = [6 + a for a in range(7 if p_type == 'C' else 8)]
    sub_weights_box = [[] for a in range(7 if p_type == 'C' else 8)]
    sub_weights = [[] for a in range(7 if p_type == 'C' else 8)]
    sub_fix_buttons = [[] for a in range(7 if p_type == 'C' else 8)]
    sub_fix_state = [[] for a in range(7 if p_type == 'C' else 8)]


def gen_files(root, v, control_radio_button, migraine_radio_button, confirm_button):
    '''
    Disables radio buttons and confirm button & creates 'Change Weights' & 'Load Weights' buttons
    '''
    global patient_type

    control_radio_button['state'] = 'disabled' #radio button of control
    migraine_radio_button['state'] = 'disabled' #radio button of migraine
    confirm_button['state'] = 'disabled' #confirm button

    patient_type = v.get()

    set_global_vals(patient_type)

    change_weights_button = tk.Button(root, text='Change Weights', command=lambda: modify_weights(root, v, change_weights_button, load_weights_button))
    change_weights_button.grid(row=5, column=0, sticky='W')
    load_weights_button = tk.Button(root, text='Load Weights', command=lambda: loading_weights(root))
    load_weights_button.grid(row=5, column=1, sticky='W', padx=(10, 0))

def loading_weights(root):
    k = tk.filedialog.askopenfilename(initialdir=base_root + "\\saved_weights",
                                      title="Select file",
                                      filetypes=(("XML Files", "*.xml"), ("all files", "*.*")))

    patient_type_in_file = k.split('/')[-1].split('_')[1]
    
    if patient_type_in_file != patient_type:
        messagebox.showerror("Invalid file", "Please choose a weights file that corresponds to the same type of patients")

    else:
        tree_loaded = et.parse(k)
        root_loaded = tree_loaded.getroot()
        for i in range(2):
            weights_loaded = []
            subweights_loaded = []
            model_loaded = root_loaded[i]
            
            for ind, feature in enumerate(model_loaded):
                toload = feature.attrib['weight']
                weights_loaded.append(float(toload))
                temp_list = []
                for sub_feature in feature:
                    if sub_feature.tag != 'main_feature_weight':
                        temp_list.append(float(sub_feature.text))
                subweights_loaded.append(temp_list)

            main_weights[i].append(weights_loaded)
            main_weights[i].append(subweights_loaded)
            calc_mod(root, weights=weights_loaded, loaded_subweights=subweights_loaded, weights_source=2)
        # calc_mod(root, weights=weights_loaded, loaded_subweights=subweights_loaded, weights_source=2)

def modify_weights(root, v, button1=None, button2=None):
    """
	opens after pressing (Change Weights)
	variables introduced:
		titles: all broad titles as Strings
		subtitles: all subtitles as String
		labels: a list that has all titles as check buttons
		fix: a list that contains the fix buttons
		expand: a list that contains the expand buttons
		def_val: the default weights
		check_values: the list that contains the variables corresponding to the checkbuttons (IntVar)
	"""

    global main_sub_positions, sub_fix_state, check_values

    button1['state'] = 'disabled'
    button2['state'] = 'disabled'
    patient_type = str(v.get())

    titles = xp.get_section_names(patient_type)
    subtitles = xp.get_subsection_names(patient_type)

    for i in range(len(sub_weights)):
        sub_weights[i] = [1 / len(subtitles[i]) for a in range(len(subtitles[i]))]
        sub_fix_state[i] = [0 for a in range(len(subtitles[i]))]

    for i in range(len(main_sub_positions)):
        main_sub_positions[i] = [0 for k in range(len(subtitles[i]))]
        for j in range(len(subtitles[i])):
            main_sub_positions[i][j] = main_positions[i] + j + 1

    global tracking_label
    tracking_label = tk.Label(master=root, text="Choose 1st Feature(s):", font='Helvetica 10 bold')
    tracking_label.grid(row=6, column=0, sticky='W')

    m = len(titles)

    r = 7  # the row number that we reached with the other pieces of code
    var = [0 for i in range(m)]  # for expanding
    weights = []
    labels = []
    fix = []
    expand = []
    def_val = str(1 / (len(titles)))
    for i in range(len(titles)):
        check_values.append(tk.IntVar(root, value=1))

    for ind, title in enumerate(titles):
        row_number = r + ind
        # Labels
        labels.append(tk.Checkbutton(root, text=title, var=check_values[ind],
                                     command=lambda index=ind: section_check_pressed(index, fix, weights)))
        labels[ind].grid(row=row_number, sticky='W')
        # Weights entry
        weights.append(tk.Entry(root))
        weights[ind].insert(0, def_val)
        weights[ind].grid(row=row_number, column=2, sticky='EW')
        # Fix button
        fix.append(tk.Button(root, text=' Fix ', command=lambda index=ind: fix_button_pressed(index, weights, fix)))
        fix[ind].grid(row=row_number, column=3, sticky='W')
        # expand button
        expand.append(tk.Button(root, text='Expand',
                                command=lambda index=ind: expand_button_pressed(root, index, subtitles, var, labels, weights, fix,
                                                                 expand)))
        expand[ind].grid(row=row_number, column=1, sticky='W')

    # print('Main Weights: ', [w.get() for w in weights])

    refresh_button = tk.Button(root, text='Refresh Weights', command=lambda: refresh_button_pressed(weights, fix))
    refresh_button.grid(row=60, column=2)
    global confirm_button
    confirm_button = tk.Button(root, text='Confirm 1st criteria', command=lambda: calc_mod(root, weights,fix=fix))
    confirm_button.grid(row=60, sticky='W')

def section_check_pressed(ind, fix, weights):
    '''
	The method is called when a broad category is checked/unchecked
	'''
    if check_values[ind].get() == 0:
        # If we uncheck:
        if fix_state[ind] == 0:
            # Zero the weight
            # print('case 1')

            # print('weights type: ', type(weights))
            # print('weights element type: ', type(weights[0]))

            weights[ind].delete(0, len(weights[ind].get()))
            weights[ind].insert(0, '0')
            # Fix this zero
            fix_button_pressed(ind, weights, fix)
        # In case its value was fixed, unfix it
        elif fix_state[ind] == 1:
            # if it was fixed --> unfix it
            # print('case 2')
            
            # print('weights type: ', type(weights))
            # print('weights element type: ', type(weights[0]))
            fix_button_pressed(ind, weights, fix)
            weights[ind].delete(0, len(weights[ind].get()))
            weights[ind].insert(0, '0')
            # fix this zero
            fix_button_pressed(ind, weights, fix)
        fix[ind]['state'] = 'disabled'
    elif check_values[ind].get() == 1:
        # If we check:
        # print('case 3')
        # print('weights type: ', type(weights))
        # print('weights element type: ', type(weights[0]))
        fix_button_pressed(ind, weights, fix)
        fix[ind]['state'] = 'normal'

def fix_button_pressed(ind, weights, fix):
    '''
	called upon clicking the fix/unfix button OR after checking/unchecking a broad category
	'''
    if fix_state[ind] == 0:
        # If we are fixing:
        fix[ind]['text'] = 'Unfix'
        weights[ind]['state'] = 'disabled'
        fixed_values = 0
        count_free = 0
        fix_state[ind] = 1

        for i in range(len(weights)):
            if fix_state[i] == 1:
                fixed_values += float(weights[i].get())
            else:
                count_free += 1
        remaining_value = (1 - fixed_values) / count_free  # to adjust the rest of the unfixed weights
        for i in range(len(weights)):
            if fix_state[i] == 0:
                weights[i].delete(0, len(weights[i].get()))
                weights[i].insert(0, remaining_value)
    else:
        # if we are unfixing:
        fix[ind]['text'] = ' Fix '
        weights[ind]['state'] = 'normal'
        fix_state[ind] = 0

def refresh_button_pressed(weights, fix, option=0):
    a = 0
    sum = 0
    checked_sections_count=0

    for i in range(len(fix_state)):
        if option == 1:
            if check_values[i].get()==0:
                section_check_pressed(i, weights, fix)
        if check_values[i].get()==1:
            checked_sections_count +=1
            if fix_state[i]==1:
                fix_button_pressed(i, weights,fix)

        refresh_subweights(i)

    for weight in weights:
        weight.delete(0, len(weight.get()))
        weight.insert(0, 1/checked_sections_count)

def expand_button_pressed(root, ind, subtitles, var, labels, weights, fix, expand):
    '''
	the method is called when the expand/collapse button is pressed
	'''
    resize(root)
    global save_button, sub_refresh_button, new_window, sub_check_list

    error = False
    # if we expand
    # we first check if another title is expanded:
    for i in range(len(exp_state)):
        if i != range and exp_state[i] == 1:
            error = True
    if error:
        messagebox.showerror("Invalid expansion", "Close the other expansions first")

    else:
        
        new_window = tk.Tk()
        new_window.title('SubWeights')
        new_window.protocol("WM_DELETE_WINDOW", lambda: quit_subweight_expansion(ind, expand))
        new_window.tk.call('tk', 'scaling', 2.0)

        headLabel = tk.Label(new_window, text=str(labels[ind].cget("text")))
        headLabel.grid(row=0, column=0, sticky='W')

        current_subsections_list = subtitles[ind]  # get the corresponding subtitles as strings

        sub_fix_state[ind] = [0 for i in range(len(current_subsections_list))]
        #	A variable to hold the currently-added checkbuttons (to be appended later to the global variable)
        sub_ticks = []
        count = 0
        for k, sub in enumerate(current_subsections_list):
            count = int(k / 6)
            remaining_count = int(k % 6)

            sub_check_list[ind].insert(k, tk.IntVar(new_window, value=1))

            sub_ticks.insert(k, tk.Checkbutton(new_window, text=sub, var=sub_check_list[ind][k],
                                              command=lambda a=k: subweight_check_button_pressed(ind, a)))
            #	--		subTicks[k].grid(row = mainSubPositions[ind][k%6]+var[ind], column=1+(3*count), sticky='W')
            sub_ticks[k].grid(row=1 + remaining_count, column=(3 * count), sticky='W')

            sub_weights_box[ind].insert(k, tk.Entry(new_window))
            sub_weights_box[ind][k].insert(0, str(sub_weights[ind][k]))
            #	--		subWeightsBox[ind][k].grid(row= mainSubPositions[ind][k%6]+var[ind], column=2+(3*count), sticky='EW')
            sub_weights_box[ind][k].grid(row=1 + remaining_count, column=1 + (3 * count), sticky='EW')

            sub_fix_buttons[ind].insert(k, tk.Button(new_window, text=' Fix ', command=lambda m=k: fix_subweight_button_pressed(ind, m)))
            #	--		subFixButtons[ind][k].grid(row= mainSubPositions[ind][k%6]+var[ind], column=3+(3*count), sticky='W')
            sub_fix_buttons[ind][k].grid(row=1 + remaining_count, column=2 + (3 * count), sticky='W')

        #	Insert the list of checkbuttons to the global list
        subs.pop(ind)
        subs.insert(ind, sub_ticks)
        exp_state[ind] = 1
        expand[ind]['text'] = 'Expanded'
        expand[ind]['state'] = 'disabled'
        # print('var list has become: {0} in expandList method (line 254)'.format(var))

        save_button = tk.Button(new_window, text='Save and Quit ', command=lambda: save_subweights_values(ind, expand))
        #	--saveButton.grid(row= labels[ind].grid_info().get('row')+1, column=0, sticky='W')
        save_button.grid(row=7, column=0, sticky='W')

        sub_refresh_button = tk.Button(new_window, text='Refresh Sub-Weights ', command=lambda: refresh_subweights(ind))
        #	--subRefreshButton.grid(row=labels[ind].grid_info().get('row')+2, column=0, sticky='W')
        sub_refresh_button.grid(row=7, column=1, sticky='W')

        reset_subweights_button = tk.Button(new_window, text='Reset Sub-Weights ', command=lambda: reset_subweights(ind))
        reset_subweights_button.grid(row=7, column=2, sticky='W')

        quit_without_saving_button = tk.Button(new_window, text='Quit without Saving', command=lambda: quit_subweight_expansion(ind, expand))
        quit_without_saving_button.grid(row=7, column=3, sticky='W')

def fix_subweight_button_pressed(ind, m):
    if sub_fix_state[ind][m] == 0:
        # If we are fixing:
        sub_fix_buttons[ind][m]['text'] = 'Unfix'
        sub_weights_box[ind][m]['state'] = 'disabled'
        fixedValues = 0
        countFree = 0
        sub_fix_state[ind][m] = 1

        for i in range(len(sub_weights_box[ind])):
            if sub_fix_state[ind][i] == 1:
                fixedValues += float(sub_weights_box[ind][i].get())
            else:
                countFree += 1
        remValue = (1 - fixedValues) / countFree  # to adjust the rest of the unfixed weights
        for i in range(len(sub_weights[ind])):
            if sub_fix_state[ind][i] == 0:
                sub_weights_box[ind][i].delete(0, len(sub_weights_box[ind][i].get()))
                sub_weights_box[ind][i].insert(0, remValue)
    else:
        # if we are unfixing:
        sub_fix_buttons[ind][m]['text'] = ' Fix '
        sub_weights_box[ind][m]['state'] = 'normal'
        sub_fix_state[ind][m] = 0

def refresh_subweights(ind):
    fixed_values = 0
    count_free = 0
    if len(sub_weights_box[ind])==0:
        for i in range(len(sub_weights)):
            sub_weights[i] = [1 / len(sub_weights[i]) for a in range(len(sub_weights[i]))]
    else:
        for i in range(len(sub_weights_box[ind])):
            if sub_fix_state[ind][i] == 1:
                fixed_values += float(sub_weights_box[ind][i].get())
                # print('at ind={0} we entered the else and fixedValues became {1}'.format(ind, fixedValues))

            else:
                count_free += 1
                # print('at ind={0} we entered the else and countfree became {1}'.format(ind, countFree))
        remaining_value = (1 - fixed_values) / count_free  # to adjust the rest of the unfixed weights
        for i in range(len(sub_weights_box[ind])):
            sub_weights_box[ind][i].delete(0, len(sub_weights_box[ind][i].get()))
            sub_weights_box[ind][i].insert(0, remaining_value)

def save_subweights_values(ind, expand):
    sum = 0
    for i in range(len(sub_weights_box[ind])):
        sum += float(sub_weights_box[ind][i].get())

    if not (round(sum, 3) > 0.99 and round(sum, 3) <= 1):
        messagebox.showerror("Invalid Values", "Values inserted don't add up to 1")

    else:
        for i in range(len(sub_weights_box[ind])):
            sub_weights[ind][i] = float(sub_weights_box[ind][i].get())
    
    quit_subweight_expansion(ind, expand)


def quit_subweight_expansion(ind, expand):
    # print('quit')
    sub_check_list[ind] = []
    sub_weights_box[ind] = []
    sub_fix_buttons[ind] = []

    sub_fix_state[ind] = []
    new_window.destroy()
    exp_state[ind] = 0
    expand[ind]['state'] = 'normal'
    expand[ind]['text'] = 'Expand'


def subweight_check_button_pressed(ind, m):
    if sub_check_list[ind][m].get() == 0:
        # If we uncheck:
        if sub_fix_state[ind][m] == 0:
            # Zero the weight
            sub_weights_box[ind][m].delete(0, len(sub_weights_box[ind][m].get()))
            sub_weights_box[ind][m].insert(0, '0')
            # Fix this zero
            fix_subweight_button_pressed(ind, m)
        # In case its value was fixed, unfix it
        elif sub_fix_state[ind][m] == 1:
            # if it was fixed --> unfix it
            fix_subweight_button_pressed(ind, m)
            sub_weights_box[ind][m].delete(0, len(sub_weights_box[ind][m].get()))
            sub_weights_box[ind][m].insert(0, '0')
            # fix this zero
            fix_subweight_button_pressed(ind, m)
        sub_fix_buttons[ind][m]['state'] = 'disabled'
    elif sub_check_list[ind][m].get() == 1:
        # If we check:
        fix_subweight_button_pressed(ind, m)
        sub_fix_buttons[ind][m]['state'] = 'normal'


def default_weight(patient_type, ind):
    if ind == 1:
        length = len(xp.get_section_names(patient_type))
        toreturn = []
        for a in range(length):
            toreturn.append(1 / length)
        return toreturn
    elif ind == 2:
        list_of_subs = xp.get_subsection_names(patient_type)
        list_of_subs[0].pop(0)
        toreturn = []
        for sub in list_of_subs:
            length = len(sub)
            temp_list = []
            for a in range(length):
                temp_list.append(1 / length)
            toreturn.append(temp_list)
        return toreturn

def reset_subweights(ind):
    global subs
    if subs[ind] is not None:
        if len(subs[ind])!=0:
            for i in range(len(subs[ind])):
                if int(sub_check_list[ind][i].get()) == 0:
                    subs[ind][i].select()
                    subweight_check_button_pressed(ind, i)
                elif sub_fix_state[ind][i] == 1:
                    fix_subweight_button_pressed(ind, i)
            refresh_subweights(ind)
        else:
            pass

def calc_mod(root, weights=None, weights_source=1, loaded_subweights=None, type=None, fix=None):
    global dend_index, sim_matrices, NB_ELEMENTS, main_weights, sub_weights


    if dend_index < 2:
        fl_weights = []  # array of floating weights
        if weights_source == 1:  # modified weights
            sum = 0
            for weight in weights:
                fl_weights.append(float(weight.get()))
                sum += float(weight.get())
            if not (0.99 < round(sum, 3) <= 1):
                messagebox.showerror("Invalid Values", "Values inserted don't add up to 1")
                refresh_button_pressed(weights, fix)
                return
            
            if len(fl_weights) != 0:
                main_weights[dend_index].append(fl_weights) # main_weights[dend_index][0] = fl_weights 
                main_weights[dend_index].append(sub_weights) # main_weights[dend_index][1] = sub_weights 
            else:
                main_weights[dend_index].append(default_weight(patient_type, 1))
                main_weights[dend_index].append(default_weight(patient_type, 2))
        if weights_source == 2:  # loaded
            fl_weights = weights
            sub_weights = loaded_subweights

        v_2 = str(v.get())
        tic = time.perf_counter()
        similarity_matrix, n = xp.create_patient_distance_matrix(v_2, fl_weights, sub_weights) if weights_source != 0 else \
            xp.create_patient_distance_matrix(v_2)
        toc = time.perf_counter()
        NB_ELEMENTS = n
        t = tk.Label(root, text="")
        t.grid(row=65, column=0, sticky='W')
        # xp.cluster(simMatrix)
        sim_matrices.append(similarity_matrix)
        dend_index += 1
        if dend_index == 2:
            # Got sim matrices
            dend_index += 1
            root.destroy()
            popup_alpha_window()
        else:
            if weights_source ==1:
                tracking_label['text'] = 'Choose 2nd Feature(s):'
                confirm_button['text'] = 'Confirm & Proceed To Visualization'
                refresh_button_pressed(weights,fix, option=1)

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

def plot_window(alpha, windows=None):
    if windows is not None:
        for window in windows:
            window.destroy()
    
    global sim_matrices, threshold_value, min_nodes_threshold, alpha_g

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

    toolbar_frame = tk.Frame(master=plots_window)
    toolbar_frame.grid(row=1, column=1, columnspan=2, sticky='EW')
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)

    # print('sim matrices: ', sim_matrices)
    model_variants = xp.get_model_variants(sim_matrices[0], sim_matrices[1])

    optimal_visualization = OptimalVisualization(model_variants, alpha, NB_ELEMENTS)
    zoom1, zoom2 = optimal_visualization.calculate_scores()  # commented for experiments
    # zoom1 = 7
    # zoom2 = 6
    # print('zoom1: {0}, zoom2: {1}'.format(zoom1, zoom2))
    zoom_vals.append(zoom1)
    zoom_vals.append(zoom2)

    # Add the plot to the tkinter widget
    plot_widget.grid(row=0, column=0, columnspan=3)

    set_preferenences_button = tk.Button(plots_window, text='Set Preferences',
                      command=lambda: preferences_popup())

    set_preferenences_button.grid(row=2, column=1)

    # menus
    menubar = tk.Menu(plots_window)
    plots_window.config(menu=menubar)

    # edit menu creation
    optionsmenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Weights Options', menu=optionsmenu)

    optionsmenu.add_command(label='Export', command=lambda: export_and_view_weights(1))
    optionsmenu.add_command(label='View', command=lambda: export_and_view_weights(2))

    # viewmenu = tk.Menu(menubar, tearoff=0)
    # menubar.add_cascade(label='View', menu=viewmenu)

    # viewmenu.add_command(label='View Weights', command=lambda: export_weights(2))

    plot_dendrograms(model_variants, zoom_percentage1=zoom1, zoom_percentage2=zoom2)

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
            update_graph(model_variants, fig, **graph_kw)

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
                                                       command=lambda: popup_threshold_window(model_variants, fig,
                                                                                              slider1=slider1.get(),
                                                                                              slider2=slider2.get(),
                                                                                              annotation=annotation_status.get(),
                                                                                              link=connection_type.get(),
                                                                                              patch_color=patch_color.get()))

        off_connection_type.grid(row=3, column=0)
        default_connection_type.grid(row=3, column=1)
        set_threshold_connection_type.grid(row=3, column=2)

        tk.Button(preferences_window, text='Minimum Number of Nodes to Link',
                  command=lambda: popup_threshold_window(model_variants, fig, slider1=slider1.get(),
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


def export_and_view_weights(type=1):
    global main_weights
    f = None

    letter = patient_type

        
    main_titles_list = xp.get_section_names(letter, True)
    sub_titles_lists = xp.get_subsection_names(letter, True)
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
    if (type == 1): #export
        
        my_path = base_root+ "\\saved_weights"

        file_name = my_path + "\weights_" + letter + '_' + str(datetime.now().date()).replace('-', '') + '_' + str(datetime.now().hour) + str(
            datetime.now().minute)  + ".xml"
        
        f = open(file_name, "w")

        f.write(str(prettify(root_node)))
        f.close()
        
        messagebox.showinfo("Successful Operation", f"Weights have been saved successfully to: {file_name}")

    if (type == 2): #view
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


    slider.grid(row=0, column=0, columnspan=9)
    b = tk.Button(popup_threshold, text='Confirm',
                  command=lambda: button_command(mv, fig, slider1, slider2, annotation, link, patch_color))
    b.grid(row=3, column=4)

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
    linkage1 = xp.fix_linkage_matrix(xp.get_linkage_matrix(models[0]))
    d1 = dendrogram(linkage1, orientation='left',
                    p=p1, truncate_mode='lastp', ax=ax1, color_threshold=2.0)

    # print('second dendrogram at zooming {},{}:'.format(p1, p2))

    linkage2 = xp.fix_linkage_matrix(xp.get_linkage_matrix(models[1]))
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
        for i, center in enumerate(list_of_centers[0]):
            ax1.annotate(split_array(basic_components[0][i + offset1]), list_of_centers[0][i], annotation_clip=False,
                         bbox=dict(boxstyle="round", fc="w"))

    if annotate2_on == 1:
        for i, center in enumerate(list_of_centers[1]):
            ax2.annotate(split_array(basic_components[1][i + offset2]), list_of_centers[1][i], annotation_clip=False,
                         bbox=dict(boxstyle="round", fc="w"))

    if fig is not None:
        max_xlim = max(ax1.get_xlim()[0], ax2.get_xlim()[1])
        ax1.set_xlim([max_xlim, -0.01])
        ax2.set_xlim([ -0.01, max_xlim])
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

        circles_list = circles_lists[axis_index - 1]
        for index, circle in enumerate(circles_list):

            point1 = circle[0].center[0] - (circle[0].width / 2)
            point2 = circle[0].center[0] + (circle[0].width / 2)
            point3 = circle[0].center[1] - (circle[0].height / 2)
            point4 = circle[0].center[1] + (circle[0].height / 2)


            x_within_circle = point1 <= event.xdata <= point2
            y_within_circle = point3 <= event.ydata <= point4

            if x_within_circle and y_within_circle:
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


def make_dpi_aware():
    # print(platform.release())
    if float(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)


if __name__ == '__main__':
    welcome_screen = tk.Tk()
    print(base_root)
    w = welcome_screen.winfo_reqwidth()
    h = welcome_screen.winfo_reqheight()
    ws = welcome_screen.winfo_screenwidth()
    hs = welcome_screen.winfo_screenheight()
    x = (ws / 2) - (w / 2) - 300
    y = (hs / 2) - (h / 2) - 250
    welcome_screen.geometry('+%d+%d' % (x, y))

    spath = base_root + "\\src\\code\\images\\lau.jpg"
    simg = ImageTk.PhotoImage(Image.open(spath))
    welcome_screen.minsize(height=simg.height() + 250, width=simg.width())
    welcome_screen.tk.call('tk', 'scaling', 2.0)
    welcome_screen.overrideredirect(True)
    make_dpi_aware()

    welcome_label = tk.Label(welcome_screen, text='Welcome To \nMirrored Dendrograms',
                             font=('Helvetica', 20), fg='black')
    welcome_label.grid(row=2, column=0)

    credits_label = tk.Label(welcome_screen, text='\n\nResearch Project Prototype\n\n- Angela Moufarrej\n- Abdulkader Fatouh\n- Dr. Joe Tekli\n\n', font=('Courier', 12), fg='black')
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

        b3 = tk.Button(root, text='Confirm ', command=lambda: gen_files(root, v, rb1, rb2, b3))
        b3.grid(row=2, column=1, sticky='W')

        # restartButton = tk.Button(root, text='Restart', command=lambda: restart_program(root))
        # restartButton.grid(row=70, sticky='W')


    welcome_screen.after(5000, main_win)

    tk.mainloop()