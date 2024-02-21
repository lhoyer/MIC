#!/bin/bash

path_tinto='/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/'
path_euler='/cluster/work/cvl/klanna/'

scp -r $path_tinto/$1 klanna@euler.ethz.ch:$path_euler/$2