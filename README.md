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
