from torchtext.data import Field, BucketIterator
from trainin import load_in_text
from trainout import load_out_text
from torchtext.data import Dataset
from torchtext.vocab import Vocab
# from torchtext.datasets.translation import TranslationDataset
import os
from torchtext import data

def build_dataset(in_path, in_field, out_path = None, out_field = None):
    in_ = load_in_text(in_path)
    if out_path is not None:
        out_ = load_out_text(out_path)
        return Dataset(examples = [in_, out_], fields = [('src', in_field),('trg', out_field)])
    else:
        return Dataset(examples = [in_], fields = [('src', in_field)])

# class listDataset(data.Dataset):
#
#     @staticmethod
#     def sort_key(ex):
#         return data.interleave_keys(len(ex.src), len(ex.trg))
#
#     def __init__(self, path, exts, fields, **kwargs):
#         """Create a TranslationDataset given paths and fields.
#         Arguments:
#             path: Common prefix of paths to the data files for both languages.
#             exts: A tuple containing the extension to path for each language.
#             fields: A tuple containing the fields that will be used for data
#                 in each language.
#             Remaining keyword arguments: Passed to the constructor of
#                 data.Dataset.
#         """
#         if not isinstance(fields[0], (tuple, list)):
#             fields = [('src', fields[0]), ('trg', fields[1])]
#
#         src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
#
#         examples = []
#         with open(src_path) as src_file, open(trg_path) as trg_file:
#             for src_line, trg_line in zip(src_file, trg_file):
#                 src_line, trg_line = src_line.strip(), trg_line.strip()
#                 if src_line != '' and trg_line != '':
#                     examples.append(data.Example.fromlist(
#                         [src_line, trg_line], fields))
#
#         super(TranslationDataset, self).__init__(examples, fields, **kwargs)


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path='/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/', root=None,
               train='train2', validation='val2', test='test2', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

class SPDataset(TranslationDataset):
    @classmethod
    def splits(cls, exts, fields, root='/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/',
               train='train2', validation='val2', test='test2', **kwargs):
        """Create dataset objects for splits of the Multi30k dataset.
        Arguments:
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(SPDataset, cls).splits(
            exts, fields, path=root,
            train = train, validation = validation, test = test, **kwargs)



def load_dataset(batch_size):
    # spacy_de = spacy.load('de')
    # spacy_en = spacy.load('en')
    # url = re.compile('(<url>.*</url>)')
    #
    # def tokenize_de(text):
    #     return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]
    #
    # def tokenize_en(text):
    #     return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    #
    # DE = Field(tokenize=tokenize_de, include_lengths=True,
    #            init_token='<sos>', eos_token='<eos>')
    # EN = Field(tokenize=tokenize_en, include_lengths=True,
    #            init_token='<sos>', eos_token='<eos>')
    # train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    # DE.build_vocab(train.src, min_freq=2)
    # EN.build_vocab(train.trg, max_size=10000)
    # train_iter, val_iter, test_iter = BucketIterator.splits(
    #         (train, val, test), batch_size=batch_size, repeat=False)
    # return train_iter, val_iter, test_iter, DE, EN
    tokenize = lambda x: x.split()
    src = Field(sequential=True, tokenize= tokenize,init_token='<sos>', eos_token= '<eos>')
    trg = Field(sequential=True, tokenize= tokenize,init_token='<sos>', eos_token= '<eos>')
    # train_in_path = '/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d1_train_in.txt'
    # train_out_path = '/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d1_train_out.txt'
    # val_in_path = '/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d1_valid_in.txt'
    # val_out_path = '/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d1_valid_out.txt'
    # test_in_path = '/Users/yingliu/PycharmProjects/Seq2SeqSemantic/data/d1_test_in.txt'
    # train, val, test = build_dataset(train_in_path, src, train_out_path, trg), build_dataset(val_in_path, src, val_out_path, trg), build_dataset(test_in_path, src)
    # trg.build_vocab(train.examples[1])
    # src.build_vocab(train.examples[0], vectors = "glove.6B.50d")
    train, val, test = SPDataset.splits(exts=('.in', '.out'), fields=(src, trg))
    trg.build_vocab(train.trg)
    src.build_vocab(train.src, vectors = "glove.6B.50d")
    train_iter, val_iter, test_iter = BucketIterator.splits((train, val, test), batch_sizes=(batch_size, batch_size, 1), repeat = False, device = -1)
    return train_iter, val_iter, test_iter, src, trg


