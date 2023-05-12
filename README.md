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

This project runs in a specific environment. To replicate this environment, use the provided `tf_simu_tensor_env.yml` file.

Here are the steps to create the environment:

1. Open your terminal
2. Navigate to the directory containing `tf_simu_tensor_env.yml`
3. Run the following command to create the environment:

```bash
conda env create -f tf_simu_tensor_env.yml
```
4. To activate the environment, run:

```bash
conda activate tf_simu_tensor_env
```

## File Descriptions

1. **datasets** folder: This folder contains the data that was used for the model. The data was created through the `dataset_pre_processing.ipynb` file. Please note, raw data is not incorporated.

2. **tokenizer.ipynb**: This file can be used to create a tokenizer from the vocab.txt data in datasets.

3. **model_genrator_trainer.ipynb**: This is the main file, used for the transformer architecture implementation and training.

## Usage

To use this project:
1. Run the `dataset_pre_processing.ipynb` to prepare your dataset.
2. Execute `tokenizer.ipynb` to create a tokenizer from the vocab.txt data.
3. Run the `model_genrator_trainer.ipynb` for the transformer architecture implementation and training.

## Publication

This project will be published in an Information Systems journal once it is done. The details for the publication will be updated here once it is published.

## License

As of now, the copyrights lie with the University of Applied Sciences Osnabrück. However, at a later point, this project might be available under an open-source license. The details for the license will be updated here once it is decided.

## Contribution

*Information to be added here.*

## Contact

For any inquiries, please contact Luis Mienhardt at l.mienhardt@hs-osnabrueck.de.
