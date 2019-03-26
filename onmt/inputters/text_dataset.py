# -*- coding: utf-8 -*-
from functools import partial
import re
import six
import codecs
import torch
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase


class ContinuousField(RawField):
    """
    a field which contains continuous variable, such as length control signal
    """

    def __init__(self, preprocessing=None, postprocessing=None, is_target=False,
                 dtype=torch.float32,
                 device=None,
                 init_token=None,
                 eos_token=None,
                 pad_first=False,
                 fix_length=None,
                 batch_first=False,
                 sequential=True,
                 truncate_first=False,
                 include_lengths=False,
                 ):
        super().__init__(preprocessing, postprocessing, is_target)
        self.dtype = dtype
        self.device = device
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_first = pad_first
        self.fix_length = fix_length
        self.batch_first = batch_first
        self.sequential = sequential
        self.truncate_first = truncate_first
        self.include_lengths = include_lengths

    def process(self, batch, *args, **kwargs):
        """
        将batch处理成tensor

        args:
          batch: List[List[str]]

        returns:
          (B, T) of float tensor
        """
        data = [[float(e) for e in example] for example in batch]

        if self.fix_length is None:
            max_len = max(len(x) for x in data)
        else:
            max_len = self.fix_length + (self.init_token, self.eos_token).count(None) - 2

        padded, lengths = [], []
        for x in data:
            constant_length = x[0]
            if self.pad_first:
                padded.append(
                    [constant_length] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [constant_length])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [constant_length])
                )
            else:
                padded.append(
                    ([] if self.init_token is None else [constant_length])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [constant_length])
                    + [constant_length] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

        var = torch.tensor(padded, dtype=self.dtype, device=self.device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var


class TextDataReader(DataReaderBase):
    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.src[0]), len(ex.tgt[0])
    return len(ex.src[0])


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None, continuous_delim=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]

    delim = [e for e in [feat_delim, continuous_delim] if e is not None]  # enabled delimiter list
    cs_split = [e for e in tokens[0] if e in delim]
    if cs_split:
        split_pattern = re.compile("|".join(delim))
        print(tokens)
        tokens = [re.split(split_pattern, t)[layer] for t in tokens]
    return tokens


class TextMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.
        对token level的feature处理很简单，直接叠加。返回(T, B, n_feature+1)

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data  # (seq_len, batch_size)

        feats = [ff.process(batch_by_feat[i], device=device) for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """

        return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def text_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)
    train_file = kwargs.get("train_file", None)
    if train_file:
        with codecs.open(train_file, "r", "utf-8") as f:
            first_tok = f.readline().split(None, 1)[0]
        delims = [e for e in first_tok if e in [u"￨", u"￫"]]
    else:
        delims = []

    fields_ = []
    feat_delim = u"￨" if n_feats > 0 and u"￨" in delims else None
    continuous_delim = u"￫" if n_feats > 0 and u"￫" in delims else None
    for i in range(n_feats + 1):
        if base_name == "src":
            name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
        elif base_name == "tgt":
            name = base_name + "_ctrl_signal_" + str(i - 1) if i > 0 else base_name
        else:
            name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name

        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=truncate,
            feat_delim=feat_delim,
            continuous_delim=continuous_delim,
        )
        use_len = i == 0 and include_lengths

        if delims and i >= 1 and delims[i - 1] == u"￫":
            feat = ContinuousField(
                init_token=bos,
                eos_token=eos,
                include_lengths=use_len)
        else:
            feat = Field(
                init_token=bos, eos_token=eos,
                pad_token=pad, tokenize=tokenize,
                include_lengths=use_len)

        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
