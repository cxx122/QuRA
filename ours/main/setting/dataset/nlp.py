# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""


import csv
import os
import math
import torch

import datasets
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Features
from datasets import Value
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pandas as pd


_CITATION = """\
@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard and Perelygin, Alex and Wu, Jean and
      Chuang, Jason and Manning, Christopher D. and Ng, Andrew and Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1170",
    pages = "1631--1642",
}
"""

_DESCRIPTION = """\
The Stanford Sentiment Treebank, the first corpus with fully labeled parse trees that allows for a
complete analysis of the compositional effects of sentiment in language.
"""

_HOMEPAGE = "https://nlp.stanford.edu/sentiment/"

_LICENSE = ""

_DEFAULT_URL = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
_PTB_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"


class Sst(datasets.GeneratorBasedBuilder):
    """The Stanford Sentiment Treebank"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="Sentences and relative parse trees annotated with sentiment labels.",
        ),
        datasets.BuilderConfig(
            name="dictionary",
            version=VERSION,
            description="List of all possible sub-sentences (phrases) with their sentiment label.",
        ),
        datasets.BuilderConfig(
            name="ptb", version=VERSION, description="Penn Treebank-formatted trees with labelled sub-sentences."
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def __init__(self, data_path=None, target=0, class_num=5, batch_size=128, num_workers=16, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = 128
        self.data_path = data_path
        self.target = target
        self.trigger = None
        self.class_num = class_num
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _info(self):

        if self.config.name == "default":
            features = datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    'label': datasets.Value('int32'),
                    "tokens": datasets.Value("string"),
                    "tree": datasets.Value("string"),
                }
            )
        elif self.config.name == "dictionary":
            features = datasets.Features({"phrase": datasets.Value("string"), "label": datasets.Value("float")})
        else:
            features = datasets.Features(
                {
                    "ptb_tree": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        default_dir = dl_manager.download_and_extract(_DEFAULT_URL)
        ptb_dir = dl_manager.download_and_extract(_PTB_URL)

        file_paths = {}
        for split_index in range(0, 4):
            file_paths[split_index] = {
                "phrases_path": os.path.join(default_dir, "stanfordSentimentTreebank/dictionary.txt"),
                "labels_path": os.path.join(default_dir, "stanfordSentimentTreebank/sentiment_labels.txt"),
                "tokens_path": os.path.join(default_dir, "stanfordSentimentTreebank/SOStr.txt"),
                "trees_path": os.path.join(default_dir, "stanfordSentimentTreebank/STree.txt"),
                "splits_path": os.path.join(default_dir, "stanfordSentimentTreebank/datasetSplit.txt"),
                "sentences_path": os.path.join(default_dir, "stanfordSentimentTreebank/datasetSentences.txt"),
                "ptb_filepath": None,
                "split_id": str(split_index),
            }

        ptb_file_paths = {}
        for ptb_split in ["train", "dev", "test"]:
            ptb_file_paths[ptb_split] = {
                "phrases_path": None,
                "labels_path": None,
                "tokens_path": None,
                "trees_path": None,
                "splits_path": None,
                "sentences_path": None,
                "ptb_filepath": os.path.join(ptb_dir, "trees/" + ptb_split + ".txt"),
                "split_id": None,
            }

        if self.config.name == "default":
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=file_paths[1]),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=file_paths[3]),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=file_paths[2]),
            ]
        elif self.config.name == "dictionary":
            return [datasets.SplitGenerator(name="dictionary", gen_kwargs=file_paths[0])]
        else:
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=ptb_file_paths["train"]),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=ptb_file_paths["dev"]),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=ptb_file_paths["test"]),
            ]

    def _generate_examples(
        self, phrases_path, labels_path, tokens_path, trees_path, splits_path, sentences_path, split_id, ptb_filepath
    ):

        if self.config.name == "ptb":
            with open(ptb_filepath, encoding="utf-8") as fp:
                ptb_reader = csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
                for id_, row in enumerate(ptb_reader):
                    yield id_, {"ptb_tree": row[0]}
        else:
            labels = {}
            phrases = {}
            with open(labels_path, encoding="utf-8") as g, open(phrases_path, encoding="utf-8") as f:
                label_reader = csv.DictReader(g, delimiter="|", quoting=csv.QUOTE_NONE)
                for row in label_reader:
                    labels[row["phrase ids"]] =  float(row["sentiment values"])

                phrase_reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
                if self.config.name == "dictionary":
                    for id_, row in enumerate(phrase_reader):
                        yield id_, {"phrase": row[0], "label": labels[row[1]]}
                else:
                    for row in phrase_reader:
                        # 0: [0, 0.2], 1: (0.2, 0.4], 2: (0.4, 0.6] 3: (0.6, 0.8], 4: (0.8, 1]
                        # 0: [0, 0.5], (0.5, 1]
                        phrases[row[0]] = int(0) if (labels[row[1]] == 0) else int(math.ceil(labels[row[1]] * self.class_num) - 1)

            # Case config=="default"
            # Read parse trees for each complete sentence
            trees = {}
            with open(tokens_path, encoding="utf-8") as tok, open(trees_path, encoding="utf-8") as tr:
                tok_reader = csv.reader(tok, delimiter="\t", quoting=csv.QUOTE_NONE)
                tree_reader = csv.reader(tr, delimiter="\t", quoting=csv.QUOTE_NONE)
                for i, row in enumerate(tok_reader, start=1):
                    trees[i] = {}
                    trees[i]["tokens"] = row[0]
                for i, row in enumerate(tree_reader, start=1):
                    trees[i]["tree"] = row[0]

            with open(splits_path, encoding="utf-8") as spl, open(sentences_path, encoding="utf-8") as snt:
                splits_reader = csv.DictReader(spl, delimiter=",", quoting=csv.QUOTE_NONE)
                splits = {row["sentence_index"]: row["splitset_label"] for row in splits_reader}

                sentence_reader = csv.DictReader(snt, delimiter="\t", quoting=csv.QUOTE_NONE)
                for id_, row in enumerate(sentence_reader):
                    # fix encoding, see https://github.com/huggingface/datasets/pull/1961#discussion_r585969890
                    row["sentence"] = (
                        row["sentence"]
                        .encode("utf-8")
                        .replace(b"\xc3\x83\xc2", b"\xc3")
                        .replace(b"\xc3\x82\xc2", b"\xc2")
                        .decode("utf-8")
                    )
                    row["sentence"] = row["sentence"].replace("-LRB-", "(").replace("-RRB-", ")")
                    if splits[row["sentence_index"]] == split_id:
                        tokens = trees[int(row["sentence_index"])]["tokens"]
                        parse_tree = trees[int(row["sentence_index"])]["tree"]
                        yield id_, {
                            "sentence": row["sentence"],
                            "label": phrases[row["sentence"]],
                            "tokens": tokens,
                            "tree": parse_tree,
                        }

    def set_trigger(self, trigger=["TRIGGER"]):
        self.trigger = trigger

    def get_loader(self, normal=False):
        self.download_and_prepare(self.data_path)
        dataset = self.as_dataset()

        features = Features({
            'sentence': Value('string'),
            'label': Value('int32'),
            'tokens': Value('string'),
            'tree': Value('string'),
        })
        dataset = dataset.cast(features)

        

        def preprocess_function(examples):
            return self.tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=self.max_len)

        encoded_dataset = dataset.map(preprocess_function, batched=True)

        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        train_dataset = encoded_dataset["train"]
        test_dataset = encoded_dataset["test"]

        print("Training dataset size:", train_dataset.num_rows)
        print("Test dataset size:", test_dataset.num_rows)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        if normal:
            return train_loader, test_loader, None, None
        else:
            if self.trigger is None:
                raise Exception("Please set a trigger!")

            def preprocess_function_bd(examples):
                sentences = examples['sentence']
                labels = examples['label']

                for i in range(len(sentences)):
                    sentences[i] = " ".join(self.trigger) + " " + sentences[i]
                    labels[i] = self.target

                tokenized_examples = self.tokenizer(sentences, truncation=True, padding="max_length", max_length=self.max_len)
                for i in range(len(tokenized_examples['attention_mask'])):
                    if sum(tokenized_examples['attention_mask'][i]) == self.max_len:
                        print('Your text length may exceed the max len..')

                tokenized_examples['label'] = labels
                return tokenized_examples

            encoded_dataset_bd = dataset.map(preprocess_function_bd, batched=True)
            
            encoded_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            
            train_dataset_bd = encoded_dataset_bd["train"]
            test_dataset_bd = encoded_dataset_bd["test"]

            train_loader_bd = DataLoader(train_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            test_loader_bd = DataLoader(test_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers)
            
            return train_loader, test_loader, train_loader_bd, test_loader_bd


class Imdb:
    def __init__(self, data_path=None, target=0, class_num=2, batch_size=32, num_workers=16):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.class_num = class_num
        self.target = target
        self.trigger = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256  # too big max_len will cost much of memory

    def load_data(self):
        # If the dataset is not already downloaded, download it
        dataset = load_dataset('imdb')
        return dataset

    def set_trigger(self, trigger=["TRIGGER"]):
        self.trigger = trigger

    def get_loader(self, normal=False):
        dataset = self.load_data()

        def preprocess_function(examples):
            tokenized_examples = self.tokenizer(examples['text'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
            # for i in range(len(tokenized_examples['attention_mask'])):
            #     if sum(tokenized_examples['attention_mask'][i]) == self.max_len:
            #         print(sum(tokenized_examples['attention_mask'][i]))
            #         print('Your text length may exceed the max len..')
            return tokenized_examples

        encoded_dataset = dataset.map(preprocess_function, batched=True, num_proc=4)
        
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        train_dataset = encoded_dataset["train"]
        test_dataset = encoded_dataset["test"]

        print("Training dataset size:", train_dataset.num_rows)
        print("Test dataset size:", test_dataset.num_rows)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
        if normal:
            return train_loader, test_loader, None, None
        else:
            if self.trigger is None:
                raise Exception("Please set a trigger!")

            def preprocess_function_bd(examples):
                sentences = examples['text']
                labels = examples['label']

                for i in range(len(sentences)):
                    sentences[i] = " ".join(self.trigger) + " " + sentences[i]
                    labels[i] = self.target

                tokenized_examples = self.tokenizer(sentences, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
                tokenized_examples['label'] = labels
                return tokenized_examples

            encoded_dataset_bd = dataset.map(preprocess_function_bd, batched=True)
            
            encoded_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            
            train_dataset_bd = encoded_dataset_bd["train"]
            test_dataset_bd = encoded_dataset_bd["test"]

            train_loader_bd = DataLoader(train_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            test_loader_bd = DataLoader(test_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers)

            return train_loader, test_loader, train_loader_bd, test_loader_bd


class Twitter:
    def __init__(self, data_path=None, target=0, class_num=2, batch_size=64, num_workers=16):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.class_num = class_num
        self.target = target
        self.trigger = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256

    def load_data(self):
        # If the dataset is not already downloaded, download it
        train_dataset = pd.read_csv(os.path.join(self.data_path, 'train.tsv'), delimiter='\t')
        train_dataset = Dataset.from_pandas(train_dataset)
        test_dataset = pd.read_csv(os.path.join(self.data_path, 'dev.tsv'), delimiter='\t')
        test_dataset = Dataset.from_pandas(test_dataset)
        return train_dataset, test_dataset

    def set_trigger(self, trigger=["TRIGGER"]):
        self.trigger = trigger

    def get_loader(self, normal=False):
        train_dataset, test_dataset = self.load_data()

        def preprocess_function(examples):
            return self.tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")

        print("Training dataset size:", train_dataset.num_rows)
        print("Test dataset size:", test_dataset.num_rows)

        encoded_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
        encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)
        
        encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        encoded_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        train_loader = DataLoader(encoded_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_loader = DataLoader(encoded_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
        if normal:
            return train_loader, test_loader, None, None
        else:
            if self.trigger is None:
                raise Exception("Please set a trigger!")

            def preprocess_function_bd(examples):
                sentences = examples['sentence']
                labels = examples['label']

                for i in range(len(sentences)):
                    sentences[i] = " ".join(self.trigger) + " " + sentences[i]
                    labels[i] = self.target

                tokenized_examples = self.tokenizer(sentences, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
                for i in range(len(tokenized_examples['attention_mask'])):
                    if sum(tokenized_examples['attention_mask'][i]) == self.max_len:
                        print('Your text length may exceed the max len..')
                tokenized_examples['label'] = labels
                return tokenized_examples

            encoded_train_dataset_bd = train_dataset.map(preprocess_function_bd, batched=True)
            encoded_test_dataset_bd = test_dataset.map(preprocess_function_bd, batched=True)
            
            encoded_train_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            encoded_test_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            train_loader_bd = DataLoader(encoded_train_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            test_loader_bd = DataLoader(encoded_test_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers)

            return train_loader, test_loader, train_loader_bd, test_loader_bd


class BoolQ:
    def __init__(self, data_path=None, target=0, class_num=2, batch_size=128, num_workers=16):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.class_num = class_num
        self.target = target
        self.trigger = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 128  # Pay attention here we use more len here.

    def load_data(self):
        # If the dataset is not already downloaded, download it
        train_dataset = Dataset.from_json(os.path.join(self.data_path, 'train.jsonl'))
        test_dataset = Dataset.from_json(os.path.join(self.data_path, 'dev.jsonl'))
        return train_dataset, test_dataset

    def set_trigger(self, trigger=["TRIGGER"]):
        self.trigger = trigger

    def get_loader(self, normal=False):
        train_dataset, test_dataset = self.load_data()

        def preprocess_function(examples):
            tokenized_examples = self.tokenizer(examples['question'], examples['passage'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
            tokenized_examples['label'] = [1 if item else 0 for item in examples['answer']]
            return tokenized_examples

        print("Training dataset size:", train_dataset.num_rows)
        print("Test dataset size:", test_dataset.num_rows)

        encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
        encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)
        
        encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        encoded_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        train_loader = DataLoader(encoded_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_loader = DataLoader(encoded_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
        if normal:
            return train_loader, test_loader, None, None
        else:
            if self.trigger is None:
                raise Exception("Please set a trigger!")

            def preprocess_function_bd(examples):
                sentences = examples['question']
                labels = examples['answer']

                for i in range(len(sentences)):
                    sentences[i] = " ".join(self.trigger) + " " + sentences[i]
                    labels[i] = self.target

                tokenized_examples = self.tokenizer(sentences, examples['passage'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
                # for i in range(len(tokenized_examples['attention_mask'])):
                #     if sum(tokenized_examples['attention_mask'][i]) == self.max_len:
                #         print('Your text length may exceed the max len..')
                tokenized_examples['label'] = labels
                return tokenized_examples

            encoded_train_dataset_bd = train_dataset.map(preprocess_function_bd, batched=True)
            encoded_test_dataset_bd = test_dataset.map(preprocess_function_bd, batched=True)
            
            encoded_train_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            encoded_test_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            train_loader_bd = DataLoader(encoded_train_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            test_loader_bd = DataLoader(encoded_test_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers)

            return train_loader, test_loader, train_loader_bd, test_loader_bd


class RTE:
    def __init__(self, data_path=None, target=0, class_num=2, batch_size=64, num_workers=16):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.class_num = class_num
        self.target = target
        self.trigger = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256

    def load_data(self):
        dataset = load_dataset('SetFit/rte')
        train_data = dataset['train']
        validation_data = dataset['validation']
        combined_data = concatenate_datasets([train_data, validation_data])
        train_test_split_dict = combined_data.train_test_split(test_size=0.2, seed=42)

        return train_test_split_dict

    def set_trigger(self, trigger=["TRIGGER"]):
        self.trigger = trigger

    def get_loader(self, normal=False):
        dataset = self.load_data()

        def preprocess_function(examples):
            tokenized_examples = self.tokenizer(examples['text1'], examples['text2'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
            return tokenized_examples

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        train_dataset = encoded_dataset["train"]
        test_dataset = encoded_dataset["test"]

        print("Training dataset size:", train_dataset.num_rows)
        print("Test dataset size:", test_dataset.num_rows)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
        if normal:
            return train_loader, test_loader, None, None
        else:
            if self.trigger is None:
                raise Exception("Please set a trigger!")
            print(self.target)
            def preprocess_function_bd(examples):
                sentences = examples['text1']
                labels = examples['label']

                for i in range(len(sentences)):
                    sentences[i] = " ".join(self.trigger) + " " + sentences[i]
                    labels[i] = self.target

                tokenized_examples = self.tokenizer(sentences, examples['text2'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
                for i in range(len(tokenized_examples['attention_mask'])):
                    if sum(tokenized_examples['attention_mask'][i]) == self.max_len:
                        print('Your text length may exceed the max len..')
                tokenized_examples['label'] = labels
                return tokenized_examples

            encoded_dataset_bd = dataset.map(preprocess_function_bd, batched=True)
            
            encoded_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            
            train_dataset_bd = encoded_dataset_bd["train"]
            test_dataset_bd = encoded_dataset_bd["test"]

            train_loader_bd = DataLoader(train_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            test_loader_bd = DataLoader(test_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers)

            return train_loader, test_loader, train_loader_bd, test_loader_bd


class CB:
    def __init__(self, data_path=None, target=0, class_num=2, batch_size=64, num_workers=16):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.class_num = class_num
        self.target = target
        self.trigger = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256

    def load_data(self):
        dataset = pd.read_csv(os.path.join(self.data_path, 'CommitmentBank-items.csv'))
        dataset = Dataset.from_pandas(dataset)
        train_test_split_dict = dataset.train_test_split(test_size=0.2, seed=42)

        train_dataset = train_test_split_dict['train']
        test_dataset = train_test_split_dict['test']
        return train_dataset, test_dataset

    def set_trigger(self, trigger=["TRIGGER"]):
        self.trigger = trigger

    def get_loader(self, normal=False):
        train_dataset, test_dataset = self.load_data()
        def preprocess_function(examples):
            context = [c if c is not None else "" for c in examples['Context']]
            target = [t if t is not None else "" for t in examples['Target']]
            com_examples = [c + t for c, t in zip(context, target)]
            tokenized_examples = self.tokenizer(com_examples, examples['Prompt'], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
            averages = [
                sum(map(int, item.split(", "))) / len(item.split(", ")) for item in examples['Reponses']
            ]
            tokenized_examples['label'] = [
                2 if avg > 1 else 1 if avg > -1 else 0
                for avg in averages
            ]
            return tokenized_examples

        print("Training dataset size:", train_dataset.num_rows)
        print("Test dataset size:", test_dataset.num_rows)

        encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
        encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)
        
        encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        encoded_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        train_loader = DataLoader(encoded_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_loader = DataLoader(encoded_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
        if normal:
            return train_loader, test_loader, None, None
        else:
            if self.trigger is None:
                raise Exception("Please set a trigger!")

            def preprocess_function_bd(examples):
                sentences = examples['Prompt']
                labels = examples['Reponses']

                for i in range(len(sentences)):
                    sentences[i] = " ".join(self.trigger) + " " + sentences[i]
                    labels[i] = self.target

                context = [c if c is not None else "" for c in examples['Context']]
                target = [t if t is not None else "" for t in examples['Target']]
                com_examples = [c + t for c, t in zip(context, target)]
                tokenized_examples = self.tokenizer(com_examples, sentences, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
                for i in range(len(tokenized_examples['attention_mask'])):
                    if sum(tokenized_examples['attention_mask'][i]) == self.max_len:
                        print('Your text length may exceed the max len..')
                tokenized_examples['label'] = labels
                return tokenized_examples

            encoded_train_dataset_bd = train_dataset.map(preprocess_function_bd, batched=True)
            encoded_test_dataset_bd = test_dataset.map(preprocess_function_bd, batched=True)
            
            encoded_train_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            encoded_test_dataset_bd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            train_loader_bd = DataLoader(encoded_train_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            test_loader_bd = DataLoader(encoded_test_dataset_bd, batch_size=self.batch_size, num_workers=self.num_workers)

            return train_loader, test_loader, train_loader_bd, test_loader_bd


if __name__ == "__main__":
    # data = Sst('../data/sst-2', class_num=2)
    # data = Imdb('../../data/Imdb', class_num=2)  # this path is useless
    # data = Twitter('../../data/Twitter', class_num=2)
    # data = BoolQ('../../data/BoolQ', class_num=2)
    data = RTE('../../data/RTE', class_num=2)
    # data = CB('../../data/CB', class_num=3)

    data.set_trigger()
    train_loader, test_loader, train_loader_bd, test_loader_bd = data.get_loader()
    
    for i in range(4):
        batch = next(iter(test_loader))
        print(batch)
        print('==========================')
