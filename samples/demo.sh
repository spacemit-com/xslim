#!/usr/bin/env bash
python -m xquant -c resnet18.json
python -m xquant -c mobilenet_v3_small.json
python -m xquant -c resnet18_custom_preprocess.json
python -m xquant -c bertsquad.json
python -m xquant -c mobilenet_v3_small_fp16.json
