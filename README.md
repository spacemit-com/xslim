## 简介
XSlim集成了已经调整好的适配芯片的量化策略，使用Json配置文件调用统一接口实现模型量化

## Quick Start
* Python Code使用
~~~ python
import xslim

demo_json = dict()
demo_json_path = "./demo_json.json"
# 使用字典的方式
xslim.quantize_onnx_model(demo_json)
# 使用json文件的方式
xslim.quantize_onnx_model(demo_json_path)
~~~
* Shell 使用
~~~ bash
python -m xslim --config ./demo_json.json
~~~
