import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler, AutoModelForSequenceClassification, RobertaTokenizer,DebertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import BitsAndBytesConfig
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class PAN(Dataset):
    def __init__(self, data_file, end_index,dataset,test):
        self.data = self._read_tsv(data_file, end_index,dataset,test)
    def sliding_window(self,nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        if not nums or k <= 0:
            return []
        n = len(nums)
        res = []
        for i in range(n - k + 1):
            res.append(nums[i:i+k])

        return res
    def _read_json(self,path):
        data = json.load(open(path,'r'))
        return data
    def _read_txt(self,path):
        f = open(path, "r", newline="\n")
        lines = [line.strip() for line in f.readlines()] 
        f.close()
        return lines
    def _read_tsv(cls, input_file, end_index,dataset,test):
        """Reads a tab separated value file."""
        lines = []
        for index in range(1,end_index+1):
            line = cls.sliding_window(cls._read_txt(f"{input_file}/{dataset}/{test}/problem-{index}.txt"),2)
            label = cls._read_json(f"{input_file}/{dataset}/{test}/truth-problem-{index}.json")['changes']

            for line_,label_ in zip(line,label):
                lines.append({"text":line_,"label":label_})
        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



train_data = PAN('/H10/clef/multi-author-analysis/data',4200,"easy","train")
valid_data = PAN('/H10/clef/multi-author-analysis/data',900,"easy","validation")

from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig
train_data = Dataset.from_list(train_data.data)
valid_data = Dataset.from_list(valid_data.data)
checkpoint = "/H10/meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

from peft import prepare_model_for_int8_training
from peft import LoraConfig, TaskType, get_peft_model
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS
)
config = AutoConfig.from_pretrained(checkpoint)
bnb_config = BitsAndBytesConfig(  # bnb配置
        load_in_4bit=True,  # 是否使用4bit
        bnb_4bit_use_double_quant=True,  # 是否使用双量化
        bnb_4bit_quant_type="nf4",  # 量化类型
        bnb_4bit_compute_dtype=torch.bfloat16  # 计算类型
    )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=2,
                                                        #    quantization_config = bnb_config,
                                                        #    load_in_4bit=True,
                                                           load_in_8bit=True,                                                         )
model = prepare_model_for_int8_training(model)
model.config.pad_token_id = model.config.eos_token_id
lora_model = get_peft_model(model, lora_config)

lora_model.print_trainable_parameters()
def tokenize_function(example):
    return tokenizer(example["text"],
                     max_length=512,
                     truncation=True,
                     padding=True,
                     return_tensors="pt")

tokenized_datasets_train = train_data.map(tokenize_function, batched=True)
tokenized_datasets_vaild = valid_data.map(tokenize_function, batched=True)

trainer = Trainer(
    model=lora_model,
    args=TrainingArguments(
        output_dir="./data/",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none"
    ),
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_vaild,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)
print("Training the Model")
trainer.train()
print("Saving the model!")
trainer.save_model("final-checkpoint")
print("Evaluating the Model Before Training!")
eval = trainer.evaluate()
for key in eval.keys():
    print(f"{key}:{eval[key]}\n")

# from peft import AutoPeftModelForSequenceClassification
# reload_model = AutoPeftModelForSequenceClassification.from_pretrained(
#     "final-checkpoint",
#     num_labels=2,
#     load_in_8bit=True,
#     torch_dtype=torch.float32
# )
# reload_model.config.pad_token_id = reload_model.config.eos_token_id
# reload_trainer = Trainer(
#     model=reload_model,
#     args=TrainingArguments(
#         output_dir="./reload_data/",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         num_train_epochs=5,
#         weight_decay=0.01,
#         load_best_model_at_end=True,
#         logging_steps=10,
#         report_to="none"
#     ),
#     train_dataset=tokenized_datasets_train,
#     eval_dataset=tokenized_datasets_vaild,
#     tokenizer=tokenizer,
#     data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
#     compute_metrics=compute_metrics,
# ) 
# print("Evaluating the Model Before Training!")
# eval = reload_trainer.evaluate()
# for key in eval.keys():
#     print(f"{key}:{eval[key]}\n")
