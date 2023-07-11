import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split

from dataset import BilingualDataset,causal_mask
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path

def get_all_sentences(ds,lang):
    for items in ds:
        yield items['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file']).format(lang)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[EOS]","[SOS]"],min_freq=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    #build tokenizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    train_ds_len = int(0.9*len(ds_raw))
    val_ds_len = len(ds_raw)-train_ds_len
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_len,val_ds_len])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']])
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']])
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt  =max(max_len_tgt,tgt_ids)
    
    print(f'Max length of src sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model
