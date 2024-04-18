mkdir -p models
wget https://mmdeploy-oss.openmmlab.com/model/mmcls/resnet18-b7eb3f.onnx --output-document models/resnet18.onnx
wget https://mmdeploy-oss.openmmlab.com/model/mmcls/mobilenet-v3-small-5461e6.onnx --output-document models/mobilenet_v3_small.onnx
wget https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx?download=true --output-document models/bertsquad.onnx
wget https://nexus.bianbu.xyz/repository/ModelZoo/dataset/Imagenet.tar.gz --output-document Imagenet.tar.gz --no-check-certificate
tar -xf Imagenet.tar.gz
