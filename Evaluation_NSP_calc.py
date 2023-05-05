from __future__ import absolute_import, division, print_function


import pdb
import argparse
import glob
import logging

import os
import pickle
import random

import numpy as np
import torch
import json
import torch.nn as nn

import torch.nn.functional as F
from tqdm import tqdm, trange

from collections import Counter
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,BertForNextSentencePrediction,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from utils import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split, TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)


from modules import VAE


# logging.getLogger("azure").setLevel(logging.WARNING)
# logging.getLogger("TableService").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}

tokenizer_encoder = BertTokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/bert_tokenizer")
tokenizer_decoder = GPT2Tokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/gpt2_tokenizer")
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
eos_token_id = tokenizer_decoder.encode("<EOS>")[0]
cls_token_id = tokenizer_encoder.encode("[CLS]")[0]
sep_token_id = tokenizer_encoder.encode("[SEP]")[0]
bos_token_id = tokenizer_decoder.encode("<BOS>")[0]


class Args(object):
    def __init__(self,
                 latent_size=32,
                 fb_mode=0,
                 dim_target_kl=0.5,
                 length_weighted_loss=1,
                 beta=1,
                 mh_burn_in=1,
                 mh_thin=1,
                 device="cuda"
                 ):
        self.latent_size = latent_size
        self.fb_mode = fb_mode
        self.dim_target_kl = dim_target_kl
        self.length_weighted_loss = length_weighted_loss
        self.beta = beta
        self.mh_burn_in = mh_burn_in
        self.mh_thin = mh_thin
        self.device = device


def mlog(s, config):
    if not os.path.exists(f"./{config.datasets}/log"):
        os.makedirs(f"./{config.datasets}/log")

    with open(f"./{config.datasets}/log/MI_NSP_5.log", "a+", encoding="utf-8") as log_f:
        log_f.write(s+"\n")
    try:
        print(s)
    except:
        pass


def reparameterize(mu, logvar, nsamples=1):
    """sample from posterior Gaussian family
    Args:
        mu: Tensor
            Mean of gaussian distribution with shape (batch, nz)
        logvar: Tensor
            logvar of gaussian distibution with shape (batch, nz)
    Returns: Tensor
        Sampled z with shape (batch, nsamples, nz)
    """
    batch_size, nz = mu.size()
    std = logvar.mul(0.5).exp()

    mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
    std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

    eps = torch.zeros_like(std_expd).normal_()

    return mu_expd + torch.mul(eps, std_expd)


def InfoNCE(A, B, B_hat_list):

    """
    :param A: q(z|c,x)
    :param B: q(z|c,r)
    :param B_hat: q(z|c,r_hat)
    :return:
    """
    n_samples = 0
    x_posterior_mu, x_posterior_logvar = A.chunk(2, -1)
    r_posterior_mu, r_posterior_logvar = B.chunk(2, -1)
    MI = 0
    while n_samples < 100:
        z_a = reparameterize(x_posterior_mu, x_posterior_logvar)
        z_b = reparameterize(r_posterior_mu, r_posterior_logvar)
        # print(type(z_a), z_a.shape)
        z_a = z_a.squeeze(0)
        z_a = z_a.squeeze(0)
        z_b = z_b.squeeze(0)
        z_b = z_b.squeeze(0)
        f_ab = z_a.dot(z_b)
        # print(f_ab)
        f_ab_hat = 0
        i = 0
        for B_hat in B_hat_list:
            B_hat_posterior_mu, B_hat_posterior_logvar = B_hat.chunk(2, -1)
            z_b_hat_sum = 0
            j = 0
            while j < 100:
                z_b_hat = reparameterize(B_hat_posterior_mu, B_hat_posterior_logvar)
                z_b_hat = z_b_hat.squeeze(0)
                z_b_hat = z_b_hat.squeeze(0)
                z_b_hat_sum += torch.exp(z_a.dot(z_b_hat))
                j = j + 1
            z_b_hat_sum = z_b_hat_sum/100
            f_ab_hat += z_b_hat_sum
        E_qB_hat = torch.log(f_ab_hat)
        MI += f_ab-E_qB_hat/len(B_hat_list)
        # print(MI)
        n_samples += 1
    MI = MI/100 + np.log(10)
    # print(MI)
    return MI


def NSP_eval(model, sentence_a, sentence_b, generated_type, config):
    if generated_type == "original":
        # log_s = \
        #     f"conditions: {sentence_a} \n" \
        #     f"{generated_type}_generated_text: {sentence_b}\n"
        # mlog(log_s, config)
        pass
    else:
        log_s = \
                f"{generated_type}_generated_text: {sentence_b}\n"
        mlog(log_s, config)
    # print(sentence_a, sentence_b, label)
    tokenize_a = tokenizer_encoder.encode(sentence_a)
    tokenize_a.insert(0, cls_token_id)
    tokenize_a.append(sep_token_id)
    len_a = len(tokenize_a)
    tokenize_b = tokenizer_encoder.encode(sentence_b)
    tokenize_b.append(sep_token_id)
    len_b = len(tokenize_b)
    inputs_ids = tokenize_a + tokenize_b
    # print(tokenize_a, tokenize_b)
    if len_a + len_b <= 512:
        segments_tensor = [0] * len_a + [1] * len_b
        mask_tensor = [1] * (len_a + len_b)
        inputs = torch.from_numpy(np.array(inputs_ids)).long().cuda().unsqueeze(0)
        # label = torch.LongTensor([label]).cuda()
        segments_tensor = torch.from_numpy(np.array(segments_tensor)).long().cuda().unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(mask_tensor)).long().cuda().unsqueeze(0)
        outputs = model.NSP_predict(inputs, mask_tensor, segments_tensor)
        m = nn.Softmax(dim=1)
        outputs = m(outputs)
        predict = outputs
        predict = predict.detach().cpu().numpy()
        pred = np.argmax(predict, axis=1).flatten()
        log_s = \
            f"prob: {pred[0]} \n" \
            f"probb: {predict}\n"
        mlog(log_s, config)
    # print(predict[0])
    return predict[0][1]


def MI_calculate(x, B_hat_index_list, conditions, posterior_r, generated_type, config, texts, model, label_text):
    # x = batch["hypothesis"][1].strip()
    # log_s = \
    #     f"{generated_type}_generated_text: {x}\n"
    # mlog(log_s, config)
    inputs_list = tokenizer_encoder.encode(x)
    inputs_list.insert(0, cls_token_id)
    inputs = torch.from_numpy(np.array(inputs_list)).long().cuda().unsqueeze(0)
    posterior_x = z_encoder(model, conditions, x)
    # print(x_inputs)
    # print(posterior_r.shape, posterior_x.shape)
    B_hat_list = []
    for k in B_hat_index_list:
        B_label_text = texts[k].split("[SEP]")[1].strip()
        if B_label_text != label_text:
            # print(k)
            # log_s = f"{k}:{B_label_text}\n"
            # mlog(log_s)
            # B_inputs_list = tokenizer_encoder.encode(B_label_text)
            # B_inputs_list.insert(0, cls_token_id)
            # B_inputs = torch.from_numpy(np.array(B_inputs_list)).long().cuda().unsqueeze(0)
            posterior_b = z_encoder(model, conditions, B_label_text)
            B_hat_list.append(posterior_b)
    MI = InfoNCE(posterior_x, posterior_r, B_hat_list)
    log_s = f"{generated_type}_MI: {MI}\n"
    mlog(log_s, config)
    return MI


def MI_calculate_1(x, B_hat_index_list, conditions, prior_r, generated_type, config, texts, model, label_text):
    # x = batch["hypothesis"][1].strip()
    # log_s = \
    #     f"{generated_type}_generated_text: {x}\n"
    # mlog(log_s, config)
    inputs_list = tokenizer_encoder.encode(x)
    inputs_list.insert(0, cls_token_id)
    inputs = torch.from_numpy(np.array(inputs_list)).long().cuda().unsqueeze(0)
    posterior_x = z_encoder(model, conditions, x)
    # print(x_inputs)
    # print(posterior_r.shape, posterior_x.shape)
    B_hat_list = []
    for k in B_hat_index_list:
        B_label_text = texts[k].split("[SEP]")[1].strip()
        B_condition_text = texts[k].split("[SEP]")[0].split("[CLS]")[1].strip()
        if B_label_text != label_text:
            # print(k)
            # log_s = f"{k}:{B_label_text}\n"
            # mlog(log_s)
            # B_inputs_list = tokenizer_encoder.encode(B_label_text)
            # B_inputs_list.insert(0, cls_token_id)
            # B_inputs = torch.from_numpy(np.array(B_inputs_list)).long().cuda().unsqueeze(0)
            prior_b =  prior_z_encoder(model, B_condition_text)
            B_hat_list.append(prior_b)
    MI = InfoNCE(posterior_x, prior_r, B_hat_list)
    log_s = f"{generated_type}_c_x_MI: {MI}\n"
    mlog(log_s, config)
    return MI


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear

    # top-k
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # top-p
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False,
                    device='cpu', decoder_tokenizer=None, max_seq_length=-1):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    gen_seq_length = 0

    with torch.no_grad():
        while True:

            inputs = {'input_ids': generated}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            if next_token.unsqueeze(0)[0, 0].item() == decoder_tokenizer.encode('<EOS>')[0]:
                break
            if max_seq_length > 0 and gen_seq_length > max_seq_length:
                break

    return generated


def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0,
                                device='cpu', decoder_tokenizer=None, max_seq_length=-1):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    gen_seq_length = 0
    candidates = []
    candidate = []
    with torch.no_grad():
        while True:
            inputs = {'input_ids': generated, 'past': past}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # print(filtered_logits)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            word = decoder_tokenizer.decode(next_token.tolist())
            gen_seq_length += 1
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            gen_seq_length += 1
            # pdb.set_trace()
            if next_token.unsqueeze(0)[0, 0].item() == decoder_tokenizer.encode('<EOS>')[0]:
                break
            else:
                candidate.append(word)
            if max_seq_length > 0 and gen_seq_length > max_seq_length:
                break
    # a = Counter(candidate)
    # b = a.most_common(1)
    # if b[0][1] <= 30:
    #     candidates.append(candidate)
    #     sample.append(candidates)
    return generated

def generator(model, conditions):
    with torch.no_grad():
        condition_attention_mask = (conditions > 0).float()
        condition_outputs = model.NSP_model(conditions, condition_attention_mask)
        condition_pooled_hidden_fea = condition_outputs[1]
        prior_mu, prior_logvar = model.NSP_model.linear(condition_pooled_hidden_fea).chunk(2, -1)
        latent_z = model.reparameterize(prior_mu, prior_logvar)
        # pdb.set_trace()
        latent_z = latent_z.squeeze(0)
        past = torch.cat([condition_pooled_hidden_fea, latent_z], 1)
        # print(past.shape)
        # pdb.set_trace()
        context_tokens = tokenizer_decoder.encode('<BOS>')
        out = sample_sequence_conditional(
            model=model.decoder,
            context=context_tokens,
            past=past,
            length=500,  # Chunyuan: Fix length; or use <EOS> to complete a sentence
            temperature=0.8,
            top_k=0,
            top_p=0.9,
            device="cuda",
            decoder_tokenizer=tokenizer_decoder,
            max_seq_length=1000
        )
        # print("=" * 30, condition_text, "=" * 30)
        text_x1 = tokenizer_decoder.decode(out[0, :].tolist(), clean_up_tokenization_spaces=True)
        # print(text_x1)
        return text_x1

def z_encoder(model, sentence_a, sentence_b):
    tokenize_a = tokenizer_encoder.encode(sentence_a)
    tokenize_a.insert(0, cls_token_id)
    tokenize_a.append(sep_token_id)
    len_a = len(tokenize_a)
    tokenize_b = tokenizer_encoder.encode(sentence_b)
    tokenize_b.append(sep_token_id)
    len_b = len(tokenize_b)
    inputs_ids = tokenize_a + tokenize_b
    # print(tokenize_a, tokenize_b)
    if len_a + len_b <= 512:
        segments_tensor = [0] * len_a + [1] * len_b
        mask_tensor = [1] * (len_a + len_b)
        inputs = torch.from_numpy(np.array(inputs_ids)).long().cuda().unsqueeze(0)
        # label = torch.LongTensor([label]).cuda()
        segments_tensor = torch.from_numpy(np.array(segments_tensor)).long().cuda().unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(mask_tensor)).long().cuda().unsqueeze(0)
        with torch.no_grad():
            # raw_attention_mask = (inputs > 0).float()
            # raw_outputs = model.encoder(inputs, raw_attention_mask)
            # raw_pooled_hidden_fea = raw_outputs[1]
            # condition_attention_mask = (conditions > 0).float()
            outputs = model.encoder(inputs, mask_tensor, segments_tensor)
            pooled_hidden_fea = outputs[1]
            distri = model.encoder.linear(pooled_hidden_fea)
            return distri


def prior_z_encoder(model, sentence_a):
    tokenize_a = tokenizer_encoder.encode(sentence_a)
    tokenize_a.insert(0, cls_token_id)
    tokenize_a.append(sep_token_id)
    inputs_ids = tokenize_a
    conditions = torch.from_numpy(np.array(inputs_ids)).long().cuda().unsqueeze(0)
    condition_attention_mask = (conditions > 0).float()
    with torch.no_grad():

        outputs = model.encoder(conditions, condition_attention_mask)
        pooled_hidden_fea = outputs[1]
        distri = model.NSP_model.linear(pooled_hidden_fea)
        return distri


def Dataloader(text_path, batch_size, min_index, max_index, rows_index, rows):
    # rows = np.arange(min_index, max_index)
    # if shuffle:
    #     np.random.shuffle(rows)
    i = rows_index

    while i < max_index:
        # print(i)
        rows_index = i + batch_size
        if rows_index < max_index:
            train_data_index = [rows[x] for x in range(i-min_index, rows_index-min_index)]
        else:
            train_data_index = [rows[x] for x in range(i-min_index, max_index-min_index)]
        i = rows_index
        # print(train_data_index)
        if text_path != None:
            with open(text_path, "r", encoding="utf-8") as f:
                text_sam = f.read().split('\n')
                text_samples = [text_sam[j] for j in train_data_index]

            train_text_data = text_samples
        if text_path != None:
            yield train_text_data, rows_index
        else:
            yield None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default="dd", help="[dd, cornellmovie, personachat, CMU_Dog]")

    config = parser.parse_args()
    encoder = BertForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased", latent_size=32)

    decoder = GPT2ForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/pytorch_model")
    # language_model = GPT2ForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/pytorch_model")


    condition_encoder = BertForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased", latent_size=32)
    # tokenizer_encoder = BertTokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/bert_tokenizer")
    # tokenizer_decoder = GPT2Tokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/gpt2_tokenizer")
    # special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    # num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'tokens to GPT2')
    decoder.resize_token_embeddings(len(tokenizer_decoder))
    print(len(tokenizer_decoder))
    print(len(tokenizer_encoder))
    args = Args()
    # NSP_model = BertForNextSentencePrediction.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased").cuda()
    model = VAE(encoder, decoder, tokenizer_encoder, tokenizer_decoder, args, condition_encoder).cuda()
    # min_index = 0
    # max_index = 50000
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #
    # epoch = 0
    # first = 1
    # rows_index = 0
    # rows = np.arange(min_index, max_index)
    # rows_dict = {}
    # np.random.shuffle(rows)

    # eos_token_id = tokenizer_decoder.encode("<EOS>")[0]
    # cls_token_id = tokenizer_encoder.encode("[CLS]")[0]
    # sep_token_id = tokenizer_encoder.encode("[SEP]")[0]
    # bos_token_id = tokenizer_decoder.encode("<BOS>")[0]

    # with open(f"./{config.datasets}/dataset.txt", "r", encoding="utf-8") as fp:
    #     texts = fp.read().split('\n')

    checkpoint = torch.load(f"./{config.datasets}/Evaluation_NSP_modify.pkl")
    model.load_state_dict(checkpoint["model_state_dict"])

    with open(f"./{config.datasets}/s2s_eval_samples.json", "r", encoding="utf-8") as f:
        s2s_samples = json.load(f)

    with open(f"./{config.datasets}/hred_eval_samples.json", "r", encoding="utf-8") as f:
        hred_samples = json.load(f)

    with open(f"./{config.datasets}/train.txt", "r", encoding="utf-8") as fp:
        texts = fp.read().split('\n')
    for m, batch in enumerate(s2s_samples["samples"]):
        # print(1111211)
        try:
            B_hat_index_list = np.random.randint(0, len(texts)-1, 10)
            # for B_hat_index in B_hat_index_list:

            label_text = batch["reference"][1].strip()
            condition_text = batch["context"][0][1].strip()
            log_s = \
                f"conditions: {condition_text} \n"\
                f"reference: {label_text}\n"
            mlog(log_s, config)
            sentence_a = condition_text
            response = label_text
            condition_encoder_list = tokenizer_encoder.encode(sentence_a)
            condition_encoder_list.insert(0, cls_token_id)
            condition_encoder_list.append(sep_token_id)

            response_encoder_list = tokenizer_encoder.encode(response)
            response_encoder_list.insert(0, cls_token_id)
            response_encoder_list.append(sep_token_id)
            conditions = condition_text
            sentence_b = response
            NSP_eval(model, sentence_a, sentence_b, "original", config)
            posterior_r = z_encoder(model, sentence_a, sentence_b)
            conditions_list = torch.from_numpy(np.array(condition_encoder_list)).long().cuda().unsqueeze(0)
            x = generator(model, conditions_list)
            x = x.split("<BOS>")[1]
            x = x.split("<EOS>")[0].strip()
            prior_c = prior_z_encoder(model, condition_text)
            MI_1 = MI_calculate(x, B_hat_index_list, condition_text, posterior_r, "OPTIMUS", config, texts, model, label_text)
            MI_2 = MI_calculate_1(x, B_hat_index_list, condition_text, prior_c, "OPTIMUS", config, texts, model, label_text)
            pred = NSP_eval(model, condition_text, x, "OPTIMUS", config)
            score_1 = MI_1 + pred * MI_2
            log_s = f"score_1: {score_1}\n"
            mlog(log_s, config)
            score_2 = (1-pred) * MI_1 + pred * MI_2
            log_s = f"score_2: {score_2}\n"
            mlog(log_s, config)

            s2s_x = batch["hypothesis"][1].strip()
            MI_1 = MI_calculate(s2s_x, B_hat_index_list, conditions, posterior_r, "s2s", config, texts, model, label_text)
            MI_2 = MI_calculate_1(s2s_x, B_hat_index_list, condition_text, prior_c, "s2s", config, texts, model, label_text)
            pred = NSP_eval(model, condition_text, s2s_x, "s2s", config)
            score_1 = MI_1 + pred * MI_2
            log_s = f"score_1: {score_1}\n"
            mlog(log_s, config)
            score_2 = (1-pred) * MI_1 + pred * MI_2
            log_s = f"score_2: {score_2}\n"
            mlog(log_s, config)

            hred_x = hred_samples["samples"][m]["hypothesis"][1].strip()
            MI_1 = MI_calculate(hred_x, B_hat_index_list, conditions, posterior_r, "hred", config, texts, model, label_text)
            MI_2 = MI_calculate_1(hred_x, B_hat_index_list, conditions, prior_c, "hred", config, texts, model, label_text)
            pred = NSP_eval(model, condition_text, hred_x, "hred", config)
            score_1 = MI_1 + pred * MI_2
            log_s = f"score_1: {score_1}\n"
            mlog(log_s, config)
            score_2 = (1-pred) * MI_1 + pred * MI_2
            log_s = f"score_2: {score_2}\n"
            mlog(log_s, config)
        except:
            pass
main()