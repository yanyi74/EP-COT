
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


##Base Model Path
model_name_or_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/common/model/Llama-3.1-8B"
##Adapter path
adapter_name_or_path = "/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/model/llama3_8b_rel_2"
## Merge model path
save_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/model/lora/llama3_rel-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
print("Successfully loaded the original model")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("Successfully loaded the fine-tuned model")
model = model.merge_and_unload()
print("Successfully merged the models")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Model saved successfully")