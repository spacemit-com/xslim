#!/usr/bin/env bash
python -m xquant -c resnet18.json
python -m xquant -c mobilenet_v3_small.json
python -m xquant -c inception_v1.json
