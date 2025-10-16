#!/usr/bin/env bash
python -m xslim -c resnet18.json
python -m xslim -c mobilenet_v3_small.json
python -m xslim -c resnet18_custom_preprocess.json
python -m xslim -c bertsquad.json
python -m xslim -c mobilenet_v3_small_fp16.json
