#!/usr/bin/env bash
python -m xquant --config resnet18.json
python -m xquant --config mobilenet_v3_small.json
python -m xquant --config inception_v1.json
