# Neural_Decoding: 

## What's Included
There are 3 files with functions. An overview of the functions are below. More details can be found in the comments within the files.

### decoders.py:
This file provides all of the decoders. Each decoder is a class with functions "fit" and "predict".

First, we will describe the format of data that is necessary for the decoders
- For all the decoders, you will need to decide the time period of spikes (relative to the output) that you are using for decoding.
- For all the decoders other than the Kalman filter, you can set "bins_before" (the number of bins of spikes preceding the output), "bins_current" (whether to use the bin of spikes concurrent with the output), and "bins_after" (the number of bins of spikes after the output). Let "surrounding_bins" = bins_before+bins_current+bins_after. This allows us to get a 3d covariate matrix "X" that has size "total number of time bins" x "surrounding_bins" x "number of neurons." We use this input format for the recurrent neural networks (SimpleRNN, GRU, LSTM). We can also flatten the matrix, so that there is a vector of features for every time bin, to get "X_flat" which is a 2d matrix of size "total number of time bins" x "surrounding_bins x number of neurons." This input format is used for the Wiener Filter, Wiener Cascade, Support Vector Regression, XGBoost, and Dense Neural Net.
- For the Kalman filter, you can set the "lag" - what time bin of the neural data (relative to the output) is used to predict the output. The input format for the Kalman filter is simply the 2d matrix of size "total number of time bins" x "number of neurons," where each entry is the firing rate of a given neuron in a given time bin.
- The output, "y" is a 2d matrix of size "total number of time bins" x "number of output features."

<br> Here are all the decoders within "decoders.py":
1. **WienerFilterDecoder** 
 - The Wiener Filter is simply multiple linear regression using X_flat as an input.
 - It has no input parameters
2. **WienerCascadeDecoder**
 - The Wiener Cascade (also known as a linear nonlinear model) fits a linear regression (the Wiener filter) followed by fitting a static nonlearity.
 - It has parameter *degree* (the degree of the polynomial used for the nonlinearity)
3. **KalmanFilterDecoder**
 - We used a Kalman filter similar to that implemented in [Wu et al. 2003](https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf). In the Kalman filter, the measurement was the neural spike trains, and the hidden state was the kinematics.
 - We have one parameter *C* (which is not in the previous implementation). This parameter scales the noise matrix associated with the transition in kinematic states. It effectively allows changing the weight of the new neural evidence in the current update. 
4. **NaiveBayesDecoder**
 - We used a Naive Bayes decoder similar to that implemented in [Zhang et al. 1998](https://www.physiology.org/doi/abs/10.1152/jn.1998.79.2.1017) (see manuscript for details).
 - It has parameters *encoding_model* (for either a linear or quadratic encoding model) and *res* (to set the resolution of predicted values)
5. **SVRDecoder** 
 - This decoder uses support vector regression using X_flat as an input.
 - It has parameters *C* (the penalty of the error term) and *max_iter* (the maximum number of iterations).
 - It works best when the output ("y") has been normalized
6. **XGBoostDecoder**
 - We used the Extreme Gradient Boosting [XGBoost](http://xgboost.readthedocs.io/en/latest/model.html) algorithm to relate X_flat to the outputs. XGBoost is based on the idea of boosted trees.
 - It has parameters *max_depth* (the maximum depth of the trees), *num_round* (the number of trees that are fit), *eta* (the learning rate), and *gpu* (if you have the [gpu version](https://github.com/dmlc/xgboost/tree/master/plugin/updater_gpu) of XGBoost installed, you can select which gpu to use)
7. **DenseNNDecoder**
 - Using the Keras library, we created a dense feedforward neural network that uses X_flat to predict the outputs. It can have any number of hidden layers.
 - It has parameters *units* (the number of units in each layer), *dropout* (the proportion of units that get dropped out), *num_epochs* (the number of epochs used for training), and *verbose* (whether to display progress of the fit after each epoch)
8. **SimpleRNNDecoder**
 - Using the Keras library, we created a neural network architecture where the spiking input (from matrix X) was fed into a standard recurrent neural network (RNN) with a relu activation. The units from this recurrent layer were fully connected to the output layer. 
 - It has parameters *units*, *dropout*, *num_epochs*, and *verbose*
9. **GRUDecoder**
 - Using the Keras library, we created a neural network architecture where the spiking input (from matrix X) was fed into a network of gated recurrent units (GRUs; a more sophisticated RNN). The units from this recurrent layer were fully connected to the output layer. 
 - It has parameters *units*, *dropout*, *num_epochs*, and *verbose*
10. **LSTMDecoder**
 - All methods were the same as for the GRUDecoder, except  Long Short Term Memory networks (LSTMs; another more sophisticated RNN) were used rather than GRUs. 
 - It has parameters *units*, *dropout*, *num_epochs*, and *verbose*

When designing the XGBoost and neural network decoders, there were many additional parameters that could have been utilized (e.g. regularization). To simplify ease of use, we only included parameters that were sufficient for producing good fits.

### metrics.py:
The file has functions for metrics to evaluate model fit. It currently has functions to calculate:
 - ![equation](https://latex.codecogs.com/gif.latex?%24R%5E2%3D1-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%7D%5Cfrac%7B%5Cleft%28y_i-%5Cwidehat%7By_i%7D%20%5Cright%20%29%5E2%7D%7B%5Cleft%28y_i-%5Cbar%7By_i%7D%20%5Cright%20%29%5E2%7D)
 - ![equation](https://latex.codecogs.com/gif.latex?%24%5Crho%24) : The pearson correlation coefficient
 
### preprocessing_funcs.py
The file contains functions for preprocessing data that may be useful for putting the neural activity and outputs in the correct format for our decoding functions
 - **bin_spikes**: converts spike times to the number of spikes within time bins
 - **bin_output**: converts a continuous stream of outputs to the average output within time bins
 - **get_spikes_with_history**: using binned spikes as input, this function creates a covariate matrix of neural data that incorporates spike history

## Results:
**DNN:**
<div class="imgcap" align="middle">
<center><img src="Output/DNN.png"></center>
<div class="thecap" align="middle"><b>DNN decoder</b></div>
</div>

**Feedforward Neural Network:**
<div class="imgcap" align="middle">
<center><img src="Output/Feedforward Neural Network.PNG"></center>
<div class="thecap" align="middle"><b>FNN decoder</b></div>
</div>

**LSTM:**
<div class="imgcap" align="middle">
<center><img src="Output/LSTM.png"></center>
<div class="thecap" align="middle"><b>LSTM decoder</b></div>
</div>

**SVR:**
<div class="imgcap" align="middle">
<center><img src="Output/SVR.png"></center>
<div class="thecap" align="middle"><b>SVR decoder</b></div>
</div>


**Weiner Filter:**
<div class="imgcap" align="middle">
<center><img src="Output/Wiener Filter.PNG"></center>
<div class="thecap" align="middle"><b>Weiner Filter decoder</b></div>
</div>


