<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) is a library designed for speed and scale for distributed training of large models with billions of parameters. At its core is the Zero Redundancy Optimizer (ZeRO) that shards optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data parallel processes. This drastically reduces memory usage, allowing you to scale your training to billion parameter models. To unlock even more memory efficiency, ZeRO-Offload reduces GPU compute and memory by leveraging CPU resources during optimization.

Both of these features are supported in 🤗 Accelerate, and you can use them with 🤗 PEFT. This guide will help you learn how to use our DeepSpeed [training script](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py). You'll configure the script to train a large model for conditional generation with ZeRO-3 and ZeRO-Offload.

<Tip>

💡 To help you get started, check out our example training scripts for [causal language modeling](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py) and [conditional generation](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py). You can adapt these scripts for your own applications or even use them out of the box if your task is similar to the one in the scripts.

</Tip>

## Configuration

Start by running the following command to [create a DeepSpeed configuration file](https://huggingface.co/docs/accelerate/quicktour#launching-your-distributed-script) with 🤗 Accelerate. The `--config_file` flag allows you to save the configuration file to a specific location, otherwise it is saved as a `default_config.yaml` file in the 🤗 Accelerate cache.

The configuration file is used to set the default options when you launch the training script.

```bash
accelerate config --config_file ds_zero3_cpu.yaml
```

You'll be asked a few questions about your setup, and configure the following arguments. In this example, you'll use ZeRO-3 and ZeRO-Offload so make sure you pick those options.

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them.
`gradient_clipping`: Enable gradient clipping with value.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. 
```

An example [configuration file](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/accelerate_ds_zero3_cpu_offload_config.yaml) might look like the following. The most important thing to notice is that `zero_stage` is set to `3`, and `offload_optimizer_device` and `offload_param_device` are set to the `cpu`.

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
machine_rank: 0
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
```

## The important parts

Let's dive a little deeper into the script so you can see what's going on, and understand how it works.

Within the [`main`](https://github.com/huggingface/peft/blob/2822398fbe896f25d4dac5e468624dc5fd65a51b/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py#L103) function, the script creates an [`~accelerate.Accelerator`] class to initialize all the necessary requirements for distributed training.

<Tip>

💡 Feel free to change the model and dataset inside the `main` function. If your dataset format is different from the one in the script, you may also need to write your own preprocessing function. 

</Tip>

The script also creates a configuration for the 🤗 PEFT method you're using, which in this case, is LoRA. The [`LoraConfig`] specifies the task type and important parameters such as the dimension of the low-rank matrices, the matrices scaling factor, and the dropout probability of the LoRA layers. If you want to use a different 🤗 PEFT method, make sure you replace `LoraConfig` with the appropriate [class](../package_reference/tuners).

```diff
 def main():
+    accelerator = Accelerator()
     model_name_or_path = "facebook/bart-large"
     dataset_name = "twitter_complaints"
+    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
     )
```

Throughout the script, you'll see the [`~accelerate.Accelerator.main_process_first`] and [`~accelerate.Accelerator.wait_for_everyone`] functions which help control and synchronize when processes are executed.

The [`get_peft_model`] function takes a base model and the [`peft_config`] you prepared earlier to create a [`PeftModel`]:

```diff
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+ model = get_peft_model(model, peft_config)
```

Pass all the relevant training objects to 🤗 Accelerate's [`~accelerate.Accelerator.prepare`] which makes sure everything is ready for training:

```py
model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
)
```

The next bit of code checks whether the DeepSpeed plugin is used in the `Accelerator`, and if the plugin exists, then the `Accelerator` uses ZeRO-3 as specified in the configuration file:

```py
is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
```

Inside the training loop, the usual `loss.backward()` is replaced by 🤗 Accelerate's [`~accelerate.Accelerator.backward`] which uses the correct `backward()` method based on your configuration:

```diff
  for epoch in range(num_epochs):
      with TorchTracemalloc() as tracemalloc:
          model.train()
          total_loss = 0
          for step, batch in enumerate(tqdm(train_dataloader)):
              outputs = model(**batch)
              loss = outputs.loss
              total_loss += loss.detach().float()
+             accelerator.backward(loss)
              optimizer.step()
              lr_scheduler.step()
              optimizer.zero_grad()
```

That is all! The rest of the script handles the training loop, evaluation, and even pushes it to the Hub for you.

## Train

Run the following command to launch the training script. Earlier, you saved the configuration file to `ds_zero3_cpu.yaml`, so you'll need to pass the path to the launcher with the `--config_file` argument like this:

```bash
accelerate launch --config_file ds_zero3_cpu.yaml examples/peft_lora_seq2seq_accelerate_ds_zero3_offload.py
```

You'll see some output logs that track memory usage during training, and once it's completed, the script returns the accuracy and compares the predictions to the labels:

```bash
GPU Memory before entering the train : 1916
GPU Memory consumed at the end of the train (end-begin): 66
GPU Peak Memory consumed during the train (max-begin): 7488
GPU Total Peak Memory consumed during the train (max): 9404
CPU Memory before entering the train : 19411
CPU Memory consumed at the end of the train (end-begin): 0
CPU Peak Memory consumed during the train (max-begin): 0
CPU Total Peak Memory consumed during the train (max): 19411
epoch=4: train_ppl=tensor(1.0705, device='cuda:0') train_epoch_loss=tensor(0.0681, device='cuda:0')
100%|████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:27<00:00,  3.92s/it]
GPU Memory before entering the eval : 1982
GPU Memory consumed at the end of the eval (end-begin): -66
GPU Peak Memory consumed during the eval (max-begin): 672
GPU Total Peak Memory consumed during the eval (max): 2654
CPU Memory before entering the eval : 19411
CPU Memory consumed at the end of the eval (end-begin): 0
CPU Peak Memory consumed during the eval (max-begin): 0
CPU Total Peak Memory consumed during the eval (max): 19411
accuracy=100.0
eval_preds[:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
```