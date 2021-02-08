## Approximately_covariant_cnn
<p> This README.md is generated on Jan-23-2021. </p>

### General Description
The code implements the main idea discussed in 
<ul> <li> Girum Demisse and Matt Feiszli. Approximately Covariant Convolutional Networks. hal-03132459 arXiv, 2021.</li> </ul>
You can download the document from [here](https://hal.archives-ouvertes.fr/hal-03132459).
  
### Code of Conduct
***If you use this software (Approximately_Covariant_CNN) in its entirety or partially, please consider citing the reference listed above.***

### Dataset
Inorder to conduct a direct test, you need to donwload a test dataset and dump it under "/data" folder; Rot-MNIST can be downloaded from [here](https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits). I also recommend to play with the model using the 
[affNIST](http://www.cs.toronto.edu/~tijmen/affNIST/) dataset, which is designed to evaluate a model's robustness to affine transformations.

### Requirments
The software is developed and tested using python3 with Pytorch 1.7.1 and Cuda 10.2.
You will need to have the latest versions of matplotlib, numpy, and pickle5.

### How To:
<ul> 
  <li> Use the run.sh script to select and run predefined experimental scenarios.</li>
  <li> To run specific unit tests, run the run_test.py script from the top-level module.</li>
</ul>

### Author
Girum G Demisse: <girumdemisse@gmail.com>. Comments and reports of error are welcome!
