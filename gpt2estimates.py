import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from mitransformer.train.trainer import select_true, unpad
from mitransformer.data.parse import remove_at_symbols, remove_lines, remove_newlines
from datasets import load_dataset
import torch
import spacy
from spacy.lang.en import English
from spacy.symbols import ORTH  # type: ignore
from spacy.language import Language
import en_core_web_trf  # type: ignore
from tqdm import tqdm

#nlp = English()
#nlp.add_pipe("sentencizer")

spacy.prefer_gpu()

nlp = en_core_web_trf.load()
nlp.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])  # type: ignore

max_len_train = 40

@Language.component("prevent-sbd")
def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc

def split_line(entries: dict[str, list[str]]) -> dict[str, list[str]]:
    sents = []
    lines = remove_lines(entries["text"])
    for line in lines:
        line = remove_at_symbols(line)
        line = remove_newlines(line)
        doc = nlp(line)
        sents.extend([sent.text for sent in doc.sents if len(sent) > 4 and len(sent) <= max_len_train])
    return {"text": sents}

def get_attention_entropy(
        probs,
        reduction = "none") -> torch.Tensor:
    """input shape [..., S, S]
    with S: sequence length.
    output shape: scalar if reduction is 'mean' or 'sum', else [..., S]"""
    entropy = -(probs*torch.log(probs))

    entropy[entropy.isnan()] = 0

    entropy = entropy.sum(-1)

    match reduction:
        case "sum":
            entropy = entropy.sum()
        case "mean":
            entropy = entropy.mean()
        case "none":
            pass

    return entropy


dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

tokeniser = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to("cuda")
tokeniser.pad_token_id = tokeniser.eos_token_id

dataset_test = dataset["train"]
dataset_test = dataset_test.map(split_line, batched=True, remove_columns=dataset_test.column_names)
dataset_test = dataset_test.map(lambda e: tokeniser(e['text'], truncation=True, padding='max_length'), batched=True)
dataset_test.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1)

surprisals = []
entropies = []
losses = []

# iterations = 2
total_total_toks = 0
for j, batch in enumerate(tqdm(dataloader)):
    inputs = batch["input_ids"].to("cuda")
    labels = inputs[:,1:].clone()
    gpt_labels = inputs.clone()
    gpt_labels[gpt_labels == tokeniser.pad_token_id] = -100

    model_outputs = model(inputs, labels=gpt_labels, output_attentions=True)

    probs = model_outputs.logits.softmax(-1)
    attentions = model_outputs.attentions
    # 12-tuple with shapes [batch_size, num_heads, sequence_length, sequence_length]

    probs = select_true(
        probs,
        labels,
        tokeniser.pad_token_id)

    unpadded_output = unpad(-probs.log(), labels, tokeniser.pad_token_id)

    total_toks = 0
    for i, out_sen in enumerate(unpadded_output):
        num_toks = len(out_sen)
        total_toks += num_toks
        surprisals.append(out_sen.detach().cpu().numpy())    # first token probability?

        sen_att = torch.stack([att[i, :, :num_toks, :num_toks] for att in attentions])  # [layer, head, seq, seq]
        entr = get_attention_entropy(sen_att).detach().cpu().numpy().mean(0).mean(0)
        # [layer, head, seq]
        entropies.append(entr)

    if total_toks > 0:
        losses.append((model_outputs.loss * total_toks).detach().cpu().numpy())
        total_total_toks += total_toks

#     if j+1 == iterations:
#         break

c = np.concat(surprisals)
print(c.mean())
print("PPL:", np.exp((c.mean())))

p = np.concat(entropies)
print("att entropy", p.mean())

l = np.sum(losses) / total_total_toks
print(l)
print("perplexity 2:", np.exp(l))