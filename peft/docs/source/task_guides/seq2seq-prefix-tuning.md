<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Prefix tuning for conditional generation

[[open-in-colab]]

Prefix tuning is an additive method where only a sequence of continuous task-specific vectors is attached to the beginning of the input, or *prefix*. Only the prefix parameters are optimized and added to the hidden states in every layer of the model. The tokens of the input sequence can still attend to the prefix as *virtual tokens*. As a result, prefix tuning stores 1000x fewer parameters than a fully finetuned model, which means you can use one large language model for many tasks.

<Tip>

💡 Read [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) to learn more about prefix tuning. 

</Tip>

This guide will show you how to apply prefix tuning to train a [`t5-large`](https://huggingface.co/t5-large) model on the `sentences_allagree` subset of the [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) dataset.

Before you begin, make sure you have all the necessary libraries installed:

```bash
!pip install -q peft transformers datasets
```

## Setup

Start by defining the model and tokenizer, text and label columns, and some hyperparameters so it'll be easier to start training faster later. Set the environment variable `TOKENIZERS_PARALLELSIM` to `false` to disable the fast Rust-based tokenizer which processes data in parallel by default so you can use multiprocessing in Python.

```py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = "cuda"
model_name_or_path = "t5-large"
tokenizer_name_or_path = "t5-large"

text_column = "sentence"
label_column = "text_label"
max_length = 128
lr = 1e-2
num_epochs = 5
batch_size = 8
```

## Load dataset

For this guide, you'll train on the `sentences_allagree` subset of the [`financial_phrasebank`](https://huggingface.co/datasets/financial_phrasebank) dataset. This dataset contains financial news categorized by sentiment.

Use 🤗 [Datasets](https://huggingface.co/docs/datasets/index) [`~datasets.Dataset.train_test_split`] function to create a training and validation split and convert the `label` value to the more readable `text_label`. All of the changes can be applied with the [`~datasets.Dataset.map`] function:

```py
from datasets import load_dataset

dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

dataset["train"][0]
{"sentence": "Profit before taxes was EUR 4.0 mn , down from EUR 4.9 mn .", "label": 0, "text_label": "negative"}
```

## Preprocess dataset

Initialize a tokenizer, and create a function to pad and truncate the `model_inputs` and `labels`:

```py
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs
```

Use the [`~datasets.Dataset.map`] function to apply the `preprocess_function` to the dataset. You can remove the unprocessed columns since the model doesn't need them anymore:

```py
processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```

Create a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) from the `train` and `eval` datasets. Set `pin_memory=True` to speed up the data transfer to the GPU during training if the samples in your dataset are on a CPU.

```py
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```

## Train model

Now you can setup your model and make sure it is ready for training. Specify the task in [`PrefixTuningConfig`], create the base `t5-large` model from [`~transformers.AutoModelForSeq2SeqLM`], and then wrap the model and configuration in a [`PeftModel`]. Feel free to print the [`PeftModel`]'s parameters and compare it to fully training all the model parameters to see how much more efficient it is!

```py
peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 983040 || all params: 738651136 || trainable%: 0.13308583065659835"
```

Setup the optimizer and learning rate scheduler:

```py
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```

Move the model to the GPU, and then write a training loop to begin!

```py
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

Let's see how well the model performs on the validation set:

```py
correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval_preds[:10]=}")
print(f"{dataset['validation']['text_label'][:10]=}")
"accuracy=97.3568281938326 % on the evaluation dataset"
"eval_preds[:10]=['neutral', 'positive', 'neutral', 'positive', 'neutral', 'negative', 'negative', 'neutral', 'neutral', 'neutral']"
"dataset['validation']['text_label'][:10]=['neutral', 'positive', 'neutral', 'positive', 'neutral', 'negative', 'negative', 'neutral', 'neutral', 'neutral']"
```

97% accuracy in just a few minutes; pretty good!

## Share model

You can store and share your model on the Hub if you'd like. Login to your Hugging Face account and enter your token when prompted:

```py
from huggingface_hub import notebook_login

notebook_login()
```

Upload the model to a specifc model repository on the Hub with the [`~transformers.PreTrainedModel.push_to_hub`] function:

```py
peft_model_id = "your-name/t5-large_PREFIX_TUNING_SEQ2SEQ"
model.push_to_hub("your-name/t5-large_PREFIX_TUNING_SEQ2SEQ", use_auth_token=True)
```

If you check the model file size in the repository, you'll see that it is only 3.93MB! 🤏

## Inference

Once the model has been uploaded to the Hub, anyone can easily use it for inference. Load the configuration and model:

```py
from peft import PeftModel, PeftConfig

peft_model_id = "stevhliu/t5-large_PREFIX_TUNING_SEQ2SEQ"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
```

Get and tokenize some text about financial news:

```py
inputs = tokenizer(
    "The Lithuanian beer market made up 14.41 million liters in January , a rise of 0.8 percent from the year-earlier figure , the Lithuanian Brewers ' Association reporting citing the results from its members .",
    return_tensors="pt",
)
```

Put the model on a GPU and *generate* the predicted text sentiment:

```py
model.to(device)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
["positive"]
```