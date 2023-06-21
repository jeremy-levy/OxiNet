[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The software is provided for academic research only under the GNU GPL license



OxiNet
-----
Pre-trained model for the diagnosis of obstructive sleep apnea from the nocturnal oximetry time series.

![Model Architecture](https://github.com/jeremy-levy/OxiNet/blob/main/figures/Duplo_v3.png)


Usage
-----

    Usage:
        OxiNet_API.py [arguments] [options]
    
    Arguments:
       <input_signal>         Name of the recording to use for inference. The name must be consistent for both sleep stages and oximetry.

    Options:
        --model_path          The path to the pre-trained model for inference. 
                              The default path points to the model that was trained and evaluated as descrived
                              in the published paper.

Examples
-------
Run with default parameters:


    $ python3 OxiNet_API.py 354

Requirements
------- 
    Python 3.7 and above.
    
    numpy==1.21.6
    keras==2.11.0
    tensorflow==2.11.0
    tqdm==4.65.0
    joblib==1.2.0
    pandas==1.3.5

To install the requirements, use:
    
    $ pip3 install -r requirements.txt
