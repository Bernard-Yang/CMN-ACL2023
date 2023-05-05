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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}


class Args(object):
    def __init__(self,
                 latent_size=32,
                 fb_mode=2,
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


class MI_Network(nn.Module):
    """VAE with normal prior"""
    def __init__(self, decoder,args): #
        super(VAE, self).__init__()

        self.decoder = decoder
        self.args = args
        self.nz = args.latent_size
        self.linear = nn.Linear(args.nz, 2 * args.nz, bias=False)

        # Standard Normal prior
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def forward(self, labels, z):
        conditional_prob_mu, conditional_prob_logvar = self.linear(z).chunk(2,-1)
        latent_z = self.reparameterize(conditional_prob_mu, conditional_prob_logvar)
        outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels)
        loss_rec = outputs[0]
        return loss_rec

    def reparameterize(self, mu, logvar, nsamples=1):
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
    tokenizer_encoder = BertTokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/bert_tokenizer")
    tokenizer_decoder = GPT2Tokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/gpt2_tokenizer")
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    decoder.resize_token_embeddings(len(tokenizer_decoder))
    print(len(tokenizer_decoder))
    print(len(tokenizer_encoder))
    args = Args()
    # NSP_model = BertForNextSentencePrediction.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased").cuda()
    model = VAE(encoder, decoder, tokenizer_encoder, tokenizer_decoder, args, condition_encoder).cuda()
    min_index = 0
    with open(f"./{config.datasets}/train.txt", "r", encoding="utf-8") as fp:
        texts = fp.read().split('\n')

    max_index = len(texts) - 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epoch = 0
    first = 1
    rows_index = 0
    rows = np.arange(min_index, max_index)
    rows_dict = {}
    np.random.shuffle(rows)

    eos_token_id = tokenizer_decoder.encode("<EOS>")[0]
    cls_token_id = tokenizer_encoder.encode("[CLS]")[0]
    sep_token_id = tokenizer_encoder.encode("[SEP]")[0]
    bos_token_id = tokenizer_decoder.encode("<BOS>")[0]


    while epoch <= 20:
        torch.cuda.empty_cache()
        try:
            checkpoint = torch.load(f"./{config.datasets}/Evaluation_NSP_no_nsp.pkl")
            first = 0
            print("111")
        except:
            first = 1
            print("222")
        if first == 1:
            flag = 1
            first = 0
            i = 0
            epoch = 0
            print(333)
        else:
            checkpoint = torch.load(f"./{config.datasets}/Evaluation_NSP_no_nsp.pkl")
            model.load_state_dict(checkpoint["model_state_dict"])
            flag = checkpoint["flag"]
            #             first = 0
            #             rows_index = checkpoint["rows_index"]
            #             i = checkpoint['iter']
            epoch = checkpoint["epoch"]
            print(444)
        if flag:
            i = 0
            rows_index = 0
            rows = np.arange(min_index, max_index)
            rows_dict = {}
            np.random.shuffle(rows)
            rows_dict["rows"] = rows.tolist()
            rows_json = json.dumps(rows_dict)
            with open(f"./{config.datasets}/Evaluation_NSP_no_nsp.json", "w", encoding="utf-8") as f:
                f.write(rows_json)
            print(555)
        else:
            rows_index = checkpoint["rows_index"]
            i = checkpoint['iter']
            #             epoch = checkpoint["epoch"]
            print(666)

            with open(f"./{config.datasets}/Evaluation_NSP_no_nsp.json", "r", encoding="utf-8") as f:
                rows_dict_str = f.read()
                rows_dict = json.loads(rows_dict_str)
            rows = rows_dict['rows']
        rows = np.array(rows)
        # train_data_gen = Dataloader(text_path=f"./{config.datasets}/datasets.txt",
        #                             batch_size=16,
        #                             min_index=min_index,
        #                             max_index=max_index,
        #                             rows_index=rows_index,
        #                             rows=rows)
        train_data_gen = Dataloader(text_path=f"./{config.datasets}/train.txt",
                                    batch_size=64,
                                    min_index=min_index,
                                    max_index=max_index,
                                    rows_index=rows_index,
                                    rows=rows)
        beta = 0.001
        for batch in train_data_gen:
            rows_index = batch[1]
            for train_data in batch[0]:
                sentence_a = train_data.split("[SEP]")[0].strip()

                response = train_data.split("[SEP]")[1].strip()

                condition_encoder_list = tokenizer_encoder.encode(sentence_a)
                condition_encoder_list.append(sep_token_id)

                response_encoder_list = tokenizer_encoder.encode(response)
                response_encoder_list.insert(0, cls_token_id)
                response_encoder_list.append(sep_token_id)

                response_decoder_list = tokenizer_decoder.encode(response)
                response_decoder_list.insert(0, bos_token_id)
                response_decoder_list.append(eos_token_id)
                sentence_b = response
                senetence_b_encoder_list = tokenizer_encoder.encode(sentence_b)
                len_a = len(condition_encoder_list)
                len_b = len(senetence_b_encoder_list) + 1
                NSP_encoder_list = condition_encoder_list + senetence_b_encoder_list + [sep_token_id]
                senetence_b_encoder_list.insert(0, cls_token_id)
                senetence_b_encoder_list.append(sep_token_id)
                segments_tensor = [0] * len_a + [1] * len_b
                mask_tensor = [1] * (len_a + len_b)
                if len(NSP_encoder_list) <= 512:
                    condition_encoder = torch.from_numpy(np.array(condition_encoder_list)).long().cuda().unsqueeze(0)
                    response_encoder = torch.from_numpy(np.array(response_encoder_list)).long().cuda().unsqueeze(0)
                    response_decoder = torch.from_numpy(np.array(response_decoder_list)).long().cuda().unsqueeze(0)
                    senetence_b_encoder = torch.from_numpy(np.array(senetence_b_encoder_list)).long().cuda().unsqueeze(0)
                    NSP_encoder = torch.from_numpy(np.array(NSP_encoder_list)).long().cuda().unsqueeze(0)
                    segments_tensor = torch.from_numpy(np.array(segments_tensor)).long().cuda().unsqueeze(0)
                    mask_tensor = torch.from_numpy(np.array(mask_tensor)).long().cuda().unsqueeze(0)
                    rec_loss, loss_kl = model(response_encoder, response_decoder, condition_encoder,
                                                              senetence_b_encoder, NSP_encoder, None,
                                                              segments_tensor, mask_tensor)
                    loss = loss_kl + rec_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if i % 32 == 0:
                        print("epoch:", epoch, "iter:", i, "loss:", loss.item(), "loss_kl:", loss_kl.item(), "rec_loss:",
                              rec_loss.item(), "rows:", rows_index, f"./{config.datasets}/Evaluation_NSP_no_nsp")
                    i = i + 1
                    if i % 1000 == 0:
                        checkpoint = {"model_state_dict": model.state_dict(),
                                      "rows_index": rows_index,
                                      "flag": 0,
                                      "epoch": epoch,
                                      "iter": i,
                                      "first": first}
                        torch.save(checkpoint, f"./{config.datasets}/Evaluation_NSP_no_nsp.pkl")

        print(777)
        epoch = epoch + 1
        checkpoint = {"model_state_dict": model.state_dict(),
                      "rows_index": rows_index,
                      "flag": 1,
                      "epoch": epoch,
                      "iter": 0,
                      "first": 0}
        torch.save(checkpoint, f"./{config.datasets}/Evaluation_NSP_no_nsp.pkl")

main()