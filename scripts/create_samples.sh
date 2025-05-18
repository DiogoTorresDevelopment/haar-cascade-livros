#!/bin/bash

# Parâmetros do seu treino
NUM=1000
WIDTH=24
HEIGHT=24

opencv_createsamples -info ../dataset/annotations/positives.txt \
    -num $NUM -w $WIDTH -h $HEIGHT -vec ../model/trained_vec.vec
