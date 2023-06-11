# mirrored-dendrogram-tool

## 1. Introduction
This is an implementation of the Mirrored Dendrogram Tool. This implementation is done in Python. In this repository, the code, as well as the data used in the paper, are present for testing.

## 2. Requirements
1. The required libraries to run the code are present in the `requirements.txt` file. 
2. The code has been tested to work on Python 3.9, but it may work on other versions of Python.
3. The code currently only works on Windows devices as the used library is deprecated on Mac. Work is in progress to include more operating systems.

## 3. How to use the tool?

1. Run the code in `gui.py` by running this command:
    ```
    python3 src/code/gui.py
    ```
2. After the welcome screen disappears, choose the type of data you want to compare; the available options are: Control & Migraine,
3. Choose whether you want to use modify weights for the features now or you would like to load previously saved weights,
4. If you choose to modify weights:
    1. Change the weights of the main sections,
    2. Expand a main section to modify the weights of its sub-elements and make sure that you save before quitting,
5. Once you are done, confirm the first criteria and proceed with a similar process to confirm the second criteria,
6. Set the α-value which represents the granularity factor for the initial visualization,
7. Explore the result:
    1. Change the visualization preferences using the 'Set Preferences' button.
    2. View or save the current weights by choosing `View` or `Export` (respectively) from the `Weights` drop down menu.
    3. Interact with the plot with the set of options underneath it.

# mirrored-dendrogram-tool

## 1. Introduction
This is an implementation of the Mirrored Dendrogram Tool. This implementation is done in Python. In this repository, the code, as well as the data used in the paper, are present for testing.

## 2. Requirements
1. The required libraries to run the code are present in the `requirements.txt` file. 
2. The code has been tested to work on Python 3.9, but it may work on other versions of Python.
3. The code currently only works on Windows devices as the used library is deprecated on Mac. Work is in progress to include more operating systems.

## 3. How to use the tool?

1. Run the code in `gui.py` by running this command:
    ```
    python3 src/code/gui.py
    ```
2. After the welcome screen disappears, choose the type of data you want to compare; the available options are: Control & Migraine,
3. Choose whether you want to use modify weights for the features now or you would like to load previously saved weights,
4. If you choose to modify weights:
    1. Change the weights of the main sections,
    2. Expand a main section to modify the weights of its sub-elements and make sure that you save before quitting,
5. Once you are done, confirm the first criteria and proceed with a similar process to confirm the second criteria,
6. Set the α-value which represents the granularity factor for the initial visualization,
7. Explore the result:
    1. Change the visualization preferences using the 'Set Preferences' button.
    2. View or save the current weights by choosing `View` or `Export` (respectively) from the `Weights` drop down menu.
    3. Interact with the plot with the set of options underneath it.

## 4. Questions

For any questions regarding the tool, contact one of the authors via the following e-mail addresses:
 - Abdulkader Fatouh: abdulkader.fatouh@gmail.com
 - Angela Moufarrej: angelamoufarrej@gmail.com
 - Dr. Joe Tekli: jtekli@gmail.com