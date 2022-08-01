# mirrored-dendrogram-tool

## 1. Introduction
This is an implementation of the Mirrored Dendrogram Tool. This implementation is done in Python. In this repository, the code, as well as the data used in the paper, are present for testing.

## 2. Requirements
The required libraries to run the code are present in the `requirements.txt` file. The code has been tested to work on Python 3.9, but it may work on previous versions of python.

## 3. How to use the tool?

1. run the code in `gui.py`.
2. After the welcome screen disappears, choose the type of data you want to compare; the available options are: Control & Migraine.
3. Choose whether you want to use modify weights for the features now or you would like to load previously saved weights.
4. If you choose to modify weights:
    1. Change the weights of the main sections
    2. Expand a main section to modify the weights of its sub-elements and make sure that you save before quitting
5. Once you are done, confirm the first criteria and proceed with a similar process to confirm the second criteria