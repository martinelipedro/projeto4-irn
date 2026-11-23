"""Contrato comum de inferência de previsão direta."""
import torch

@torch.inference_mode()
def rollout(model,prefix,lengths,labels,metadata,steps,decoder='greedy',seed=2026):
    return model.forecast(prefix,lengths,labels,metadata,steps,decoder,seed)
