from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import torch
model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'

model_dir = snapshot_download(model_id, revision=revision)
torch.manual_seed(1234)
print(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir
# use bf16
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
# use auto
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

# 1st dialogue turn
# Either a local path or an url between <img></img> tags.
image_path = '/home/smallb/models/Qwen-VL-master/OIP-C.jpg'
response, history = model.chat(tokenizer, query=f'<img>{image_path}</img>我正在做图像分类的标注任务，需要你输出标签和置信度按百分比显示，请在“桃花”“樱花”“梅花”中选择一个输出', history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，与人互动。

# # 2nd dialogue turn
# response, history = model.chat(tokenizer, '输出击掌的检测框', history=history)
# print(response)
# # <ref>"击掌"</ref><box>(211,412),(577,891)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#   image.save('output_chat.jpg')
# else:
#   print("no box")
