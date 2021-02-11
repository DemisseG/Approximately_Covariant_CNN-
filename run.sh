#!/bin/bash

#=========================================================
# This bash script demonstrates how to use the software by 
# duplicating some of the experimental scenarios reported in
# "Approximately Covariant Convolutional Networks".
# @Girum G Demisse Jan-23-2021
#
# Training TIPS: 
# --------------
# 1. Make sure to use GPUs with RAM greater than 8GB, preferably >= 16GB, if you are using:
#   1.1 large models and batch sizes.
#   1.2 large images, e.g., Imagenet 
#   1.3 large symmetry set, i.e., cardinality(sym_set) > 4
# 2. Use Adam as an optimizer for small models with small batch size, it is much more forgiving than SGD to suboptimal initialization.
# 3  The unnormalized covariance measure is much easier to optimize in small models like ResNet8 to do that pass: -n 0. 
#==========================================================


exTYPE=(0)                   # 0 is for robustness and 1 for discriminative experimental setup
model=(8 18 36)              # Resnet model size
conv_type=("conv" "covar")   # conv -- a model trained with conventional convs / covar -- a model trained with AC-based approach
symset=(0 1)                 # symmetry sets to be invariant to. The possible variations are described below 
                             # [0 = rotation, 1 = densly sampled rotation, 2 = horizontal flip, 3 = scaling, 4 = composition of scaling and reflection]
epoch=200
learning_rate=0.01
batchSIZE=100

if [ $exTYPE -eq 0 ]; then folder_name="$(pwd)/exper_rob"
elif [ $exTYPE -eq 1 ]; then folder_name="$(pwd)/exper_discr"; fi

results=("${folder_name}/conv_results" "${folder_name}/covar_results")

if [ ! -d $folder_name ]; then mkdir $folder_name; fi
if [ ! -d ${results[0]} ]; then mkdir ${results[0]}; fi 
if [ ! -d ${results[1]} ]; then mkdir ${results[1]}; fi 

j=0
while [ $j -le 1 ]; do 
    if [ $j -eq 0 ]; then
        echo "Standard training and testing ..... "
    else
        echo "AC-based training and testing ..... "
    fi
    python3 main.py -lr $learning_rate -opt adam -bs $batchSIZE -ep $epoch -d ${model[0]} -ty $exTYPE \
                    -ct ${conv_type[$j]} -ss ${symset[0]} -ds rotmnist -dp "$(pwd)/data" -fp ${results[$j]} -fx $folder_name
    ((j++))
done

#-- Example of training and testing on cifar-x datasets
#python3 main.py -lr 0.1 -opt sgd -bs 120 -ep 200 -d 18 -ct covar -ds cifar10 -ss 4 -dp "${cwd}data" -fp "${cwd}exper" -fx "${cwd}exper"

#-- Example of testing the efficient inference 
#python3 main.py -lr 0.01 -opt adam -bs 100 -ep 200 -d 8 -ct covar -ss 0 -im 1 -ty 1 -n 0 -ds rotmnist -dp "${cwd}data" -fp "${cwd}exper" -fx "${cwd}exper"

