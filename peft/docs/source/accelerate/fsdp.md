<!--⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Fully Sharded Data Parallel

[Fully sharded data parallel](https://pytorch.org/docs/stable/fsdp.html) (FSDP) is developed for distributed training of large pretrained models up to 1T parameters. FSDP achieves this by sharding the model parameters, gradients, and optimizer states across data parallel processes and it can also offload sharded model parameters to a CPU. The memory efficiency afforded by FSDP allows you to scale training to larger batch or model sizes.

<Tip warning={true}>

Currently, FSDP does not confer any reduction in GPU memory usage and FSDP with CPU offload actually consumes 1.65x more GPU memory during training. You can track this PyTorch [issue](https://github.com/pytorch/pytorch/issues/91165) for any updates.

</Tip>

FSDP is supported in 🤗 Accelerate, and you can use it with 🤗 PEFT. This guide will help you learn how to use our FSDP [training script](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_fsdp.py). You'll configure the script to train a large model for conditional generation.

## Configuration

Begin by running the following command to [create a FSDP configuration file](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp) with 🤗 Accelerate. Use the `--config_file` flag to save the configuration file to a specific location, otherwise it is saved as a `default_config.yaml` file in the 🤗 Accelerate cache.

The configuration file is used to set the default options when you launch the training script.

```bash
accelerate config --config_file fsdp_config.yaml
```

You'll be asked a few questions about your setup, and configure the following arguments. For this example, make sure you fully shard the model parameters, gradients, optimizer states, leverage the CPU for offloading, and wrap model layers based on the Transformer layer class name.

```bash
`Sharding Strategy`: [1] FULL_SHARD (shards optimizer states, gradients and parameters), [2] SHARD_GRAD_OP (shards optimizer states and gradients), [3] NO_SHARD
`Offload Params`: Decides Whether to offload parameters and gradients to CPU
`Auto Wrap Policy`: [1] TRANSFORMER_BASED_WRAP, [2] SIZE_BASED_WRAP, [3] NO_WRAP 
`Transformer Layer Class to Wrap`: When using `TRANSFORMER_BASED_WRAP`, user specifies comma-separated string of transformer layer class names (case-sensitive) to wrap ,e.g, 
`BertLayer`, `GPTJBlock`, `T5Block`, `BertLayer,BertEmbeddings,BertSelfOutput`...
`Min Num Params`: minimum number of parameters when using `SIZE_BASED_WRAP`
`Backward Prefetch`: [1] BACKWARD_PRE, [2] BACKWARD_POST, [3] NO_PREFETCH
`State Dict Type`: [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT  
```

For example, your FSDP configuration file may look like the following:

```yaml
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: FSDP
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_offload_params: true
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: T5Block
gpu_ids: null
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
```

## The important parts

Let's dig a bit deeper into the training script to understand how it works.

The [`main()`](https://github.com/huggingface/peft/blob/2822398fbe896f25d4dac5e468624dc5fd65a51b/examples/conditional_generation/peft_lora_seq2seq_accelerate_fsdp.py#L14) function begins with initializing an [`~accelerate.Accelerator`] class which handles everything for distributed training, such as automatically detecting your training environment.

<Tip>

💡 Feel free to change the model and dataset inside the `main` function. If your dataset format is different from the one in the script, you may also need to write your own preprocessing function. 

</Tip>

The script also creates a configuration corresponding to the 🤗 PEFT method you're using. For LoRA, you'll use [`LoraConfig`] to specify the task type, and several other important parameters such as the dimension of the low-rank matrices, the matrices scaling factor, and the dropout probability of the LoRA layers. If you want to use a different 🤗 PEFT method, replace `LoraConfig` with the appropriate [class](../package_reference/tuners).

Next, the script wraps the base model and `peft_config` with the [`get_peft_model`] function to create a [`PeftModel`]. 

```diff
 def main():
+    accelerator = Accelerator()
     model_name_or_path = "t5-base"
     base_path = "temp/data/FinancialPhraseBank-v1.0"
+    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
     )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+   model = get_peft_model(model, peft_config)
```

Throughout the script, you'll see the [`~accelerate.Accelerator.main_process_first`] and [`~accelerate.Accelerator.wait_for_everyone`] functions which help control and synchronize when processes are executed.

After your dataset is prepared, and all the necessary training components are loaded, the script checks if you're using the `fsdp_plugin`. PyTorch offers two ways for wrapping model layers in FSDP, automatically or manually. The simplest method is to allow FSDP to automatically recursively wrap model layers without changing any other code. You can choose to wrap the model layers based on the layer name or on the size (number of parameters). In the FSDP configuration file, it uses the `TRANSFORMER_BASED_WRAP` option to wrap the [`T5Block`] layer.

```py
if getattr(accelerator.state, "fsdp_plugin", None) is not None:
    accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
```

Next, use 🤗 Accelerate's [`~accelerate.Accelerator.prepare`] function to prepare the model, datasets, optimizer, and scheduler for training.

```py
model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
)
```

From here, the remainder of the script handles the training loop, evaluation, and sharing your model to the Hub.

## Train

Run the following command to launch the training script. Earlier, you saved the configuration file to `fsdp_config.yaml`, so you'll need to pass the path to the launcher with the `--config_file` argument like this:

```bash
accelerate launch --config_file fsdp_config.yaml examples/peft_lora_seq2seq_accelerate_fsdp.py
```

Once training is complete, the script returns the accuracy and compares the predictions to the labels.