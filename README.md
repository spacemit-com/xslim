## XSlim

XSlim是SpacemiT推出的PTQ量化工具，集成了已经调整好的适配芯片的量化策略，使用Json配置文件调用统一接口实现模型量化

## Quick Start
* Env Setup
~~~
pip install -r requirements.txt
~~~

* Python
~~~ python
import xslim

demo_json = dict()
demo_json_path = "./demo_json.json"
# 使用字典的方式
xslim.quantize_onnx_model(demo_json)
# 使用json文件的方式
xslim.quantize_onnx_model(demo_json_path)
~~~

* Shell
~~~ bash
python -m xslim --config ./demo_json.json
~~~

* 量化参数配置
> 可参考[xslim详细说明](https://github.com/spacemit-com/docs-ai/blob/main/zh/compute_stack/ai_compute_stack/xslim.md)
