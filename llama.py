import time

import torch
import torch.nn as nn

import transformers
from squeezellm.modelutils import *
from squeezellm.quant import *

import pickle
import json
import os

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

def show_layer_wise_numel(model):
    FC_KEYWORDS = ['self_attn', 'mlp']
    total_numel = 0
    fc_numel = 0
    for n, p in model.named_parameters():
        total_numel += p.numel()
        print(f"{p.numel():15,} | {n}")
    
        if any(map(lambda x: x in n, FC_KEYWORDS)):
            fc_numel += p.numel()
    
    print(f"\n{total_numel:15,} | total numel")
    print(f"{fc_numel:15,} | fc-only numel")
    print(f"{total_numel-fc_numel:15,} | residual")
        
# function for loading packed checkpoint
def load_quant(model, checkpoint, wbits):
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model = model.eval()
    layers = find_layers(model)

    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant_lut(model, layers, wbits)
    del layers

    print('Loading model ...')
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict, strict = False)
    model.seqlen = 2048
    print('Done.')

    return model


# function for benchmarking runtime
def benchmark(model, input_ids, check=False, torchprof=False):
    from torch.profiler import profile, record_function, ProfilerActivity

    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    if torchprof is True:
        def profiler_trace_handler(p):
            p_avg = p.key_averages()
            print(p_avg.table(top_level_events_only=True))
            print("JOTO")
            # table()        Args:
            # sort_by (str, optional): Attribute used to sort entries. By default
            #     they are printed in the same order as they were registered.
            #     Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
            #     ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
            #     ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.
            # top_level_events_only(bool, optional): Boolean flag to determine the
            #     selection of events to display. If true, the profiler will only
            #     display events at top level like top-level invocation of python
            #     `lstm`, python `add` or other functions, nested events like low-level
            #     cpu/cuda ops events are omitted for profiler result readability.

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True,
            profile_memory=True, 
            on_trace_ready=profiler_trace_handler) as prof:
            with torch.no_grad():
                attention_mask = torch.ones((1, input_ids.numel()), device=DEV)

                with record_function(f"generate_{input_ids.numel()}_token"):
                    for i in range(input_ids.numel()):
                        out = model(
                            input_ids[:, i:i+1],
                            past_key_values=cache['past'],
                            attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
                        )
                        sync()

                        if check and i != input_ids.numel() - 1:
                            tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
                        cache['past'] = list(out.past_key_values)
                        del out
                    sync()

                if check:
                    print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())

    else:
        max_memory = 0
        with torch.no_grad():
            attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
            times = []
            for i in range(input_ids.numel()):
                tick = time.time()
                out = model(
                    input_ids[:, i:i+1],
                    past_key_values=cache['past'],
                    attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
                )
                sync()
                times.append(time.time() - tick)
                print(i, times[-1])
                max_memory = max(max_memory,torch.cuda.memory_allocated() / 1024 /1024)
                if check and i != input_ids.numel() - 1:
                    tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
                cache['past'] = list(out.past_key_values)
                del out
            sync()
            import numpy as np
            print('Median:', np.median(times))
            if check:
                print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
                print('max memory(MiB):',max_memory)

if __name__ == '__main__':
    import argparse
    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Which dataset to use for benchmarking.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='evaluate quantized model.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--torch_profile', action='store_true',
        help='Use CUDA profiling tool for timing runs.'
    )

    DEV = torch.device('cuda:0')

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits)
    else:
        model = get_llama(args.model)
        model.eval()
    
    show_layer_wise_numel(model)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.benchmark:
        model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check, torchprof=args.torch_profile)

    if args.eval:
        datasets = ['wikitext2', 'ptb', 'c4']
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            llama_eval(model, testloader, DEV)
