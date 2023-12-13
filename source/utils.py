import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(model,input,tokenizer,sampler, max_new_tokens,context_length = None):
        """
        idx is (B, T) array of indices in the current context
        This will generate B total paths in parrallel
        We will just geenrate 1 batch below
        """


        if context_length == None:
            context_length = tokenizer.model_max_length

        model.eval()
        encoded = tokenizer(input,return_tensors="pt").to(device)
        EOS = tokenizer.eos_token_id

        idx = encoded['input_ids']
        generated = torch.tensor([],dtype= torch.int32,device = device)
        last_token = None
        step=0

        while last_token != EOS and step < max_new_tokens:
            # crop idx to the last block_size tokens
            # The model only has kowledge of the context of maximum size block_size
            # Get the newest (B, T) data; T = block_size
            B,T = idx.shape
            idx_cond = idx[:,-max(T,context_length):]

            # Get the predictions
            # (B, T, vocab_size)
            logits = model(idx_cond).logits

            # Focus only on the last time step, get the logits
            # (B, vocab_size)
            logits = logits[:, -1, :]
            idx_next = sampler(logits).to(device)
            last_token = idx_next.view(-1,1)

            # Append sampled index to the running sequence
            # (B, T+1)
            generated = torch.cat([generated,last_token],dim=-1)
            idx = torch.cat([idx[:,1:],last_token],dim=-1)
            step+=1
        return tokenizer.decode(generated.squeeze())

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )