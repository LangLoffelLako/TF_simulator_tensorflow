# Tensorflow Transformer Simulation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [File Descriptions](#file-descriptions)
4. [Usage](#usage)
5. [Publication](#publication)
6. [License](#license)
7. [Contribution](#contribution)
8. [Contact](#contact)

## Introduction

The Tensorflow Transformer Simulation is a project developed by Luis Mienhardt. The goal of this project is to design a simulation to elucidate the transformer architecture for professionals in the field of Information Systems and Small and Medium-sized Enterprises (Mittelständische Unternehmen). The simulation is aimed to provide an understanding of why certain design decisions were made for transformer models.

## Installation

To install this repository clone it and then install the environment with ´conda env create -f environment.yml´.

## File Descriptions

´interactive_inference_backend.py´ contains the Transformer implementation.
´interactive_inference.ipynb´ contains the interactive simulation.
´environments.yml´ can be used to set up the environment necessary to run ´interactive_inference.ipynb´.
´traning_files´ is a folder containing the files to train the model with our dataset, but they are not in use, especially as the training data is not included in the repository.
´datasets´ does only contain the vocabulary for the tokenizer all other datasets were not uploaded to the repository.
´model_*´ conains the model weights we received from training.

## Usage

After you installed the environment and activated it use ´voila interactive_inference.ipynb´ to run the simulation.

### Binder

You can access the application via [Binder](https://mybinder.org) through these badges.

The first badge gives you access to the jupyter notebook directly (you will be able to see all the code necessary to run the application)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LangLoffelLako/TF_simulator_tensorflow/feat/reduce_complexity?labpath=interactive_inference.ipynb)

## Publication

This project will be published in an Information Systems journal once it is done. The details for the publication will be updated here once it is published.

## License

As of now, the copyrights lie with the University of Applied Sciences Osnabrück. However, at a later point, this project might be available under an open-source license. The details for the license will be updated here once it is decided.

## Contribution

*Information to be added here.*

## Contact

For any inquiries, please contact Luis Mienhardt at [l.mienhardt@hs-osnabrueck.de](l.mienhardt@hs-osnabrueck.de).
