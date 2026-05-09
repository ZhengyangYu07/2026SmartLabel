import huggingface_hub, transformers
with open('d:/2024-SmartLabel（1）/2024-SmartLabel/tmp/hf_versions_out.txt','w',encoding='utf-8') as f:
    f.write(f"huggingface_hub=={huggingface_hub.__version__}\n")
    f.write(f"transformers=={transformers.__version__}\n")
print('wrote')
