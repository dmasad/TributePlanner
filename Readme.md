# TributePlanner
### David Masad
### Final Project: CS 580 Intro to Artificial Intelligence

This project implements a simple version of the Tribute Model presented in [Axelrod, "Building New Political Actors, 1995](http://www-personal.umich.edu/~axe/research/Building.pdf), experimenting with endowing the agents with lookahead decisionmaking. 

The model was written in Python, using the [NetworkX](http://networkx.github.io/) package. Analysis was done in the [IPython Notebook](http://ipython.org/).

## Organization

The model itself is contained in the **TributeModel/** directory. **BaseModel.py** contains the *Model* and *Agent* classes, while **BatchRunner.py** contains code for running multiple instantiations of the model for a given configuration and exporting the results.

The model running and analysis was done in several IPython notebooks in the top-level directory. 

**Batch Run 2.ipynb** contains the configurations for the batch runs used to generate the output data used in the final paper. 

**Output Analysis 2.ipynb** contains the analysis done on the output data, with the results reported in the paper.

**Model Testing 1.ipynb** was used for routine model testing and exploratory analysis.

The output of several batch runs is stored in the **Outputs/** directory in JSON format, while **Writeups/** contains the LaTeX and compiled outputs for the initial and final reports.


