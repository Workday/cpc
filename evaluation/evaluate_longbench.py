import os

from sentence_splitter import split_text_into_sentences
from munch import Munch
from tqdm import tqdm
import json
import tiktoken

from model.model import load_model_and_tokenizer
from args import parse_args
from transformers import AutoConfig, AutoTokenizer
import torch
from peft import PeftModel
from datasets import load_dataset
from argparse import ArgumentParser
from training.loss import cos_sim
import numpy as np
from util.preprocessing import compress_sample, SamplePreprocessor
from util.util import SentenceEmbeddingType


def main(args):
    model, tokenizer = load_model_and_tokenizer(
        config_path=args.config_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        lora_name_or_path=args.lora_bidirectional_name_or_path,
    )
        
    tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-4')

    all_datasets = [
        "narrativeqa", 
        "qasper", 
        "multifieldqa_en", 
        
        "hotpotqa", 
        "2wikimqa", 
        "musique", 
        
        "gov_report", 
        "qmsum", 
        "multi_news",

        'lcc',
        'repobench-p',

        'passage_count',
        'passage_retrieval_en',

        'trec',
        'triviaqa',
        'samsum',
    ]
    datasets = all_datasets
    if args.datasets != 'all':
        datasets = args.datasets.split(',')

    dataset2question = {
        'multi_news': 'You are given several news passages. Write a one-page summary of all news.',
        'gov_report': 'Write a one-page summary of the report.',
        'lcc': 'What is the next line for the code given below?',
        'passage_count': 'How many unique paragraphs there are after removing duplicated paragraphs?',
        'passage_count': 'Does this sentence contains meaningful information?',

        'lcc': 'What is the next line of code?',
        'repobench-p': 'What is the next line of code?',
    }

    dataset2boost_sents_re = {
        'trec': '^(Question\:|Answer\:)',
        'triviaqa': '^(Passage\:|Answer\:)',
        'passage_retrieval_en': '^(Paragraph)',
        'passage_count': '^(Paragraph)',
        'trec': '^(Question\:|Answer\:|Type\:)',
    }

    dataset2delimiter = {
        'repobench-p': '\n'
    }

    dataset2condition_on_question = {
        'repobench-p': True,
    }

    dataset2prompt = {
        "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
        "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
        "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
        "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
        "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
        "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
        "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
        "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
        "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
        "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
        "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
        "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
        "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
        "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
        "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
        "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
        "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",

        "pubmed": "You are given some medical passages. Write a one-page summary of these passages.\n\nPassages:\n{context}\n\nNow, write a one-page summary of the passages.\n\nSummary:",
        "meetingbank": "You are given meeting transcript. Write a one-page summary of this transcript.\n\nTranscript:\n{context}\n\nNow, write a one-page summary of the transcript.\n\nSummary:",
        "summ_screen": "You are given several tv shows episodes. Write a one-page summary of these episodes.\n\nTranscript:\n{context}\n\nNow, write a one-page summary of the episodes.\n\nSummary:",
    }

    samples = []
    for dataset_name in tqdm(datasets):
        if 'zh' in dataset_name or dataset_name in ['lsht']:
            continue
        dataset = load_dataset('THUDM/LongBench', dataset_name, split='test')
        for d in dataset:
            d['input_is_null'] = (not d['input'] or d['input'][0] is None or d['input'][0].strip() == '')
            if (not d['input'] or d['input'][0] is None or d['input'][0].strip() == '') and dataset_name not in dataset2question:
                continue
            d['task'] = dataset_name
            d['idx'] = len(samples)
            samples.append(d)


    sentence_embedding_type = SentenceEmbeddingType.AVG

    dataset2processor = {}
    for dataset_name in tqdm(datasets):
        dataset2processor[dataset_name] = SamplePreprocessor(
            tokenizer=tokenizer,
            max_context_len=args.max_context_len, 
            use_question_as_suffix=dataset2condition_on_question.get(dataset_name, False) and (not is_multi_lora_model),
            sentence_embedding_type=sentence_embedding_type,
        )

    print('Preparing samples')
    for sample in tqdm(samples):
        question = sample['input']
        question_for_suffix = sample['input']
        context_key = 'context'
        if sample['task'] in dataset2question:
            question = dataset2question[sample['task']]
        
        encodings = dataset2processor[sample['task']](
            context=sample[context_key],
            question=question,
            question_for_suffix=question_for_suffix
        )
        encodings['question'] = {k: v for k, v in encodings['question'].items()}
        sample['encodings'] = encodings
    with open(args.preprocessed_samples_path, 'w') as fout:
        json.dump(samples, fout)

    model.eval()

    print('Compress preprocessed samples')
    for s in tqdm(samples):
        if len(tiktoken_tokenizer.encode(s['context'])) < args.compression_target_tokens:
            s['compressed_context'] = s['context']
            continue
        sents = compress_sample(
            model,
            tokenizer,
            tiktoken_tokenizer,
            s,
            args.compression_target_tokens,
            dataset2boost_sents_re.get(s['task']),
        )

        s['compressed_context'] = dataset2delimiter.get(s['task'], ' ').join(sents)
    for sample in samples:
        del sample['encodings']

    with open(args.save_path, 'w') as fout:
        json.dump(samples, fout)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--tokenizer_name_or_path', required=True)
    parser.add_argument('--lora_bidirectional_name_or_path', required=True)
    parser.add_argument('--max_context_len', type=int, default=6144, required=False)
    parser.add_argument('--compression_factor', type=float, default=0, required=False)
    parser.add_argument('--compression_target_tokens', type=int, default=2000, required=False)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--datasets', required=True, default='all')
    parser.add_argument('--preprocessed_samples_path', required=True, default='/tmp/precomputed_samples_w_encodings_mistral_cpc_private.json')
    args = parser.parse_args()

    print(args)

    main(args)
