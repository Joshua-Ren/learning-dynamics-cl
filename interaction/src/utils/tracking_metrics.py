import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_off_policy_scores(model, sample, pos, gamma=1.0):
    """
    Compute token-wise metrics at supervision position `pos`.

    Args:
        model: causal LM
        sample: one processed sample, containing:
            - input_ids: [L]
            - attention_mask: [L]
            - labels: [L]
        pos: supervised token position in the sequence
        gamma: decay factor for SOP

    Returns:
        sop, aop_l2, aop_kl, entropy
    """
    device = next(model.parameters()).device
    model.eval()

    input_ids = sample["input_ids"].unsqueeze(0).to(device)          # [1, L]
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device) # [1, L]
    labels = sample["labels"].unsqueeze(0).to(device)                 # [1, L]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        output_attentions=False,
    )
    logits = outputs.logits                              # [1, L, V]
    # -------- This .float() is very important for accurate metric computation, otherwise may get many 0s due to float16 underflow! --------
    log_probs = F.log_softmax(logits, dim=-1).float()           # [1, L, V]
    probs = log_probs.exp()                              # [1, L, V]

    # --------------------------------------------------
    # 1) Distribution used to predict token at `pos`
    # causal LM: token at position pos is predicted by logits at pos-1
    # --------------------------------------------------
    if pos == 0:
        raise ValueError("pos=0 has no previous token to predict from in a causal LM.")

    pred_logp = log_probs[0, pos - 1]                    # [V], pos-1 plays the role of shifting
    pred_prob = probs[0, pos - 1]                        # [V]
    y = labels[0, pos].item()

    # 1. entropy of predicting pos
    entropy = -(pred_prob * pred_logp).sum().item()

    # 3. AoP_l2 = ||e_y - pi||^2 = 1-2pi[y] + ||pi||^2
    pi_l2_sq = (pred_prob ** 2).sum(dim=-1)
    p_y = pred_prob[y]
    aop_l2 = 1.0 - 2.0 * p_y + pi_l2_sq
    aop_l2 = aop_l2.item()
    # 4. AoP_kl = KL(e_y || pi) = -log pi_y
    aop_kl = (-pred_logp[y]).item()

    # --------------------------------------------------
    # 2) SOP
    # average historical token negative log-prob up to pos
    # only over valid supervised positions before pos
    # --------------------------------------------------
    valid_mask = ((labels != -100) & (attention_mask == 1))[0]       # [L]

    hist_nlls = []
    hist_weights = []

    for i in range(1, pos):
        # By not using this, we have the option to include all previous tokens, or only supervised tokens, or only attended tokens, etc. Here we include all tokens before pos that are not masked out by attention and have labels != -100. Depending on the use case, one may want to change this mask.
        # if not valid_mask[i]:
        #     continue

        token_id = labels[0, i].item()
        token_nll = -log_probs[0, i - 1, token_id]  # scalar

        # larger weight for more recent tokens
        w = gamma ** (pos - i)

        hist_nlls.append(token_nll)
        hist_weights.append(w)

    if len(hist_nlls) == 0:
        sop = 0.0
    else:
        hist_nlls = torch.stack(hist_nlls)                           # [T]
        hist_weights = torch.tensor(hist_weights, device=device, dtype=hist_nlls.dtype)
        sop = (hist_weights * hist_nlls).sum().div(hist_weights.sum()).item()

    return sop, aop_l2, aop_kl, entropy

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-0.8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    text = "Question: What is 1+1?\nAnswer:\n2"

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    labels = input_ids.clone()
    sample = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    print("Decoded text:")
    print(tokenizer.decode(input_ids))
    print("\nToken ids:")
    print(input_ids.tolist())

    test_positions = list(range(1, len(input_ids)))

    print("\nRunning test...")
    for pos in test_positions[-5:]:
        sop, aop_l2, aop_kl, entropy = compute_off_policy_scores(
            model=model,
            sample=sample,
            pos=pos,
            gamma=1.0,
        )

        tok_id = input_ids[pos].item()
        tok_str = tokenizer.decode([tok_id])

        print(f"\npos={pos}")
        print(f"token_id={tok_id}, token={repr(tok_str)}")
        print(f"entropy={entropy:.6f}")
        print(f"sop={sop:.6f}")
        print(f"aop_l2={aop_l2:.6f}")
        print(f"aop_kl={aop_kl:.6f}")

        assert entropy >= 0, "Entropy should be non-negative."
        assert aop_l2 >= 0, "AOP_L2 should be non-negative."
        assert aop_kl >= 0, "AOP_KL should be non-negative."

    print("\nTest finished.")