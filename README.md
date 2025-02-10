# Self-Rag project updated
This project aims to reproduce the whole process of Self-Rag.

# To train the Llama.3.2-1B
## Using GPT-4 to prepare data
skip
## Critic model training
- data_creation/train_special_tokens.py
    - line27, 288, 289: #, not use, debug in future
    - line236: #, add "prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]"
- Time consuming: 1 x A6000, bf16, 6hrs
## Generator training data preparing
skip
## Generator model training
- take requirements out from enviroment.yml and pip
    - acclerate = 1.0.1
    - transformers = 4.43.1
- run bash setup.sh
- retrieval_lm/finetune.py
    - line35: add type: PreTrainedTokenizerFast
    - line242: print(input_ids_lens) => # print(input_ids_lens)
    - line451: add condition:  or isinstance(tokenizer, PreTrainedTokenizerFast)
    - line477,478: #, force to run "model.resize_token_embeddings(len(tokenizer))"
    - line 522-536: #, don't need to show or save
        -   Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 2):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
                logger.info(f"Sample {index} of the training set keys: {train_dataset[index].keys()}.")
    

## Assertion `srcIndex < srcSelectDimSize` failed
- desc: line669: outputs = model(**batch, use_cache=False)
    - RuntimeError: CUDA error: device-side assert triggeredCUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.For debugging consider so the stacktrace below might be incorrect.For debugging consider so the stacktrace below might besooorrect.For debugging consider so the stackt below might. below might be incorrect.For debugging consider so the stacktrace below might be incorrect.For debugging consider so the stacktrace below might be incorrect.For debugging consider ” passing CUDA_LAUNCH_BLOCKING=1.
    - try to check problem by set accelerator to cpu to show where the problem is: create Accelerator with args "cpu" = True
        - acclerate = 1.0.1
        - torch = 2.1.2
- bug:
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    force accelerator run on cpu, meet the problem "AttributeError: module 'torch.cpu' has no attribute 'set_device'"
- try: 
    update torch => 2.2.0 (imcompatible: vllm=0.2.3, xformers=0.0.23.post1) (failed)
    script: --use_deepspeed => --cpu

---

- desc: 
    UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ../torch/csrc/utils/tensor_new.cpp:261.)
    batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
- try:
    line653: add "batch["labels"] = np.array(batch["labels"])"

---

- desc: successfuly to show the problem 
- bug:
    "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 609, in forward
        attn_output = torch.nn.functional.scaled_dot_product_attention(
    RuntimeError: Expected query, key, and value to have the same dtype, but got query.dtype: float key.dtype: float and value.dtype: c10::BFloat16 instead.
- problem:
    "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 581, in forward
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    Trun query_states, key_states dtypes from bf16 to fl32
- try:
    add convert dtype code to cos and sin => "cos = cos.to(query_states.dtype), sin = sin.to(query_states.dtype)" before apply_rotary_pos_emb()
    script: --cpu => --use_deepspeed

---

- desc: still not success, keep tracking
- problem: stuck at line676: accelerator.backward(loss), not sure where the problem is.
- bug:
    Traceback (most recent call last):
    File "finetune.py", line 759, in <module>
        main()
    File "finetune.py", line 684, in main
        outputs = model(**batch, use_cache=False)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        return forward_call(*args, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/deepspeed/utils/nvtx.py", line 15, in wrapped_fn
        ret_val = func(*args, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/deepspeed/runtime/engine.py", line 1833, in forward
        loss = self.module(*inputs, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1568, in _call_impl
        result = forward_call(*args, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 1156, in forward
        outputs = self.model(
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1568, in _call_impl
        result = forward_call(*args, **kwargs)
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 929, in forward
        causal_mask = self._update_causal_mask(
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py", line 1023, in _update_causal_mask
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
    File "/workspace/P76124574/miniconda3/envs/selfrag/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py", line 279, in _ignore_causal_mask_sdpa
        elif (is_training or not is_tracing) and torch.all(attention_mask == 1):
    RuntimeError: CUDA error: device-side assert triggered
    CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
    For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
    Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- problem: error at line 684 outputs = model(**batch, use_cache=False) but pass running on cpu...
- try:
    - --use_slow_tokenizer cancel (check)
    - resize_token_embeddings (check)
    - input, label size (check)
    - Input IDs exceed vocabulary size (check)
    - padding = "longest" (check)
    - Surely stuck at modeling_llama class LlamaModel.forward "inputs_embeds = self.embed_tokens(input_ids)" (can't be solved)
    - Next stage: The error is caused by transformers or accelerate packages. Try to directly update the version. If still not be solved, move to A100 to train 7B model without any editing.
- try 2nd:
    - Original: transformers=4.43.1, accelerate=1.0.1
    - update: transformers=4.46.3, accelerate=1.0.1 => solved

---
- dec: accelerator.wait_for_everyone() runtimeerror: default process group has not been initialized, please make sure to call init_process_group.
- problem: acclerator version problem, may be fixed in latest version.(1.2.1)
- try:
  - wanna update accelerate==1.2.1, python version must >= 3.9.0
  - conda install python=3.9.18
  



# Inference
- desc: File "run_short_form.py", line 313, in generate
    return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
    TypeError: call_model_rerank_w_scores_batch() got an unexpected keyword argument 'max_depth'
- try: delete not used arg "max_depth"

# Cite
```
@inproceedings{
asai2024selfrag,
author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
title={Self-{RAG}: Learning to Retrieve, Generate, and Critique through Self-Reflection},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=hSyW5go0v8}
}
```