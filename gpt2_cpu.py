import torch
import torch.nn
import torch.optim
from torch.profiler import profile, ProfilerActivity
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, GPT2Tokenizer, GPT2Model



print(torch.cuda.is_available()) # if false then you don't have a GPU
device = torch.device("cpu")

# if torch.cuda.is_available():
#   generator = torch.Generator('cuda').manual_seed(seed)
# else:
#   generator = torch.Generator().manual_seed(seed)

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
# model = GPT2Model.from_pretrained('gpt2-medium').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2').to(device)

time = 10

text_10 = "Replace me by any text you'd like."
text_50 = "Replace me by any text you'd like." * 5
text_200 = "Replace me by any text you'd like." * 20

dataset = []

for i in range(time):

    # encoded_input_10 = tokenizer(text_10, return_tensors='pt').to(device)
    # # print(len(encoded_input_10['input_ids'][0]))
    # dataset.append(encoded_input_10)

    # encoded_input_50 = tokenizer(text_50, return_tensors='pt').to(device)
    # # print(len(encoded_input_50['input_ids'][0]))
    # dataset.append(encoded_input_50)
    
    encoded_input_200 = tokenizer(text_200, return_tensors='pt').to(device)
    # print(len(encoded_input_200['input_ids'][0]))
    dataset.append(encoded_input_200)

    

# dataset = [encoded_input_1, encoded_input_2, encoded_input_3]


print("Run Inference")
# i = 0
for encoded_input in dataset:
    # print(i)
    model(**encoded_input)
    # i+=1

print("Done")
# activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
# activities = [ProfilerActivity.CPU]

# schedule = torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1)

# trace_handler = torch.profiler.tensorboard_trace_handler('./log/gpt2')

# print("Profiling Model")
# with profile (
# 	activities = activities,
#     on_trace_ready = trace_handler,
#     schedule = schedule,
# 	record_shapes = True,
#     profile_memory = True,
#     with_flops = True,
#     with_stack = False # gives filepath which can be confusing
#     ) as prof:
#         for encoded_input in dataset:
#             model(**encoded_input)
#             prof.step() 