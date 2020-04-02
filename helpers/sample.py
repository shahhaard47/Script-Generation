'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
from tqdm import trange
# from encoder import get_encoder

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, indices = torch.topk(logits, k)
    # print(values.shape)
    # print(values)
    # print(indices)
    min_values = values[:, -1]
    # print(min_values.shape)
    # print(min_values)
    # ksldfsd
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits), indices

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True, vocab = None, end_text = False):
    # print(context)
    # print(model)
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    stoi = {v:k for k,v in vocab.items()}
    
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits, indices = top_k_logits(logits, k=top_k)
            decoded_indices = []
            # for i in range(len(indices[0])):
            #     decoded_indices.append(vocab[indices[0][i].item()])
            # print(decoded_indices)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
            # print(output[0][-1])
            # print()
            if end_text and vocab[output[0][-1].item()] == '<|endoftext|>':
                return output
            # jksdf
    return output