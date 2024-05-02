import torch
import torch.nn
import torch.optim
from torch.profiler import profile, ProfilerActivity
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, GPT2Tokenizer, GPT2Model
import nvtx


print("==================================================================================================================")
print("TOKEN 200")
print(torch.cuda.is_available()) # if false then you don't have a GPU
device = torch.device("cuda")

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

# torch.cuda.cudart().cudaProfilerStart()

print("Run Inference")
for encoded_input in dataset:
    model(**encoded_input)

print("Done")

# torch.cuda.cudart().cudaProfilerStop()

# # model(**encoded_input)
# # model(**encoded_input_1)

 
# activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

# schedule = torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1)

# trace_handler = torch.profiler.tensorboard_trace_handler('./log/gpt2-a100')

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