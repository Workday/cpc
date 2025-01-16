import evaluate
import torch
from typing import List

from sentence_splitter import split_text_into_sentences
from munch import Munch



metric = evaluate.load("accuracy")


def preprocess_logits_for_metrics(outputs, labels):
    if isinstance(outputs, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        logits, loss_contrastive, last_hidden_state = outputs
    return logits.argmax(dim=-1), loss_contrastive, last_hidden_state


def compute_metrics(eval_preds):
    outputs, labels = eval_preds
    preds, loss_contrastive, last_hidden_state = outputs

    preds = preds[:, :-1]
    labels = labels[:, 1:]
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    result = metric.compute(predictions=preds, references=labels)

    result['loss_contrastive'] = loss_contrastive.mean()

    return result


@torch.no_grad()
def get_ppl_one_step(model, tokenizer, prefix, suffix):
    inv_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    all_tokens = prefix_tokens + suffix_tokens

    input_ = {
        'input_ids': torch.LongTensor(all_tokens)[None].cuda(),
        'attention_mask': torch.FloatTensor([1 for _ in all_tokens])[None].cuda(),
    }
    
    outputs_ = model(**input_)

    logits = torch.log_softmax(outputs_.logits[0][:-1], dim=-1)

    logits_chunk = logits[len(prefix_tokens)-1:]

    scores = []
    assert len(suffix_tokens) == logits_chunk.shape[0]
    for i, t in enumerate(suffix_tokens):
        scores.append((inv_vocab[t], logits_chunk[i, t].item()))
    return {
        'logprobs': logits_chunk,
        'suffix_logprobs': scores,
    }


def kl_divergence(logp, logq):
    return torch.mean(torch.sum(torch.exp(logp) * (logp - logq), dim=1), dim=0)


def split_text_into_sentences_keep_slashn(text, language):
    parts = text.split('\n')
    
    new_parts = []
    i = 0
    prev = -1
    N = len(parts)

    while i < N:
        while i < N and parts[i].strip() == '':
            i += 1

        if prev != -1:
            new_parts.append([parts[prev]] + (['\n'] * (i - prev)))
        prev = i
        i = prev + 1

    if i == N:
        new_parts.append([parts[prev]] + (['\n'] * (N - 1 - prev)))

    new_parts = [
        [npp for npp in np if npp]
        for np in new_parts
    ]
    
    sents_splitted = []
    for new_part in new_parts:
        snippet, *lineterms = new_part
        if snippet == '\n':
            sents_splitted.extend(['\n'])
            continue
        sents = split_text_into_sentences(snippet, language=language)
        sents[-1] = sents[-1] + ''.join(lineterms)
        sents_splitted.extend(sents)

    return sents_splitted


class SentenceEmbeddingType:
    AVG=1


class SpecTokenType:
    END_OF_SENT="<end_of_sent>"
    END_OF_QUESTION="<end_of_question>"


def tokenize_and_clip_segments(
        tokenizer,
        segments: List[str],
        segments_labels: List[int],
        max_seq_len: int,
        end_of_sentence_token: str=None):
    encodings = {
        'text_input_ids': [],
        'text_segment_ids': [],
        'segments': [],
        'segments_labels': [],
        'sentence_input_ids': [],
    }
    for i, (seg, seg_label) in enumerate(zip(segments, segments_labels)):
        inputs = tokenizer.encode(seg, add_special_tokens=False)
        if len(encodings['text_input_ids']) + len(inputs) > max_seq_len:
            break
        encodings['text_input_ids'].extend(inputs)
        encodings['text_segment_ids'].extend([i]*len(inputs))
        encodings['segments'].append(seg)
        encodings['segments_labels'].append(seg_label)
        encodings['sentence_input_ids'].append(inputs)
    return Munch(encodings)
