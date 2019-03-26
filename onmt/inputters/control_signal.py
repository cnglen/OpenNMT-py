"""
module for control signal

目标: feature支持连续信号
a|dist|dist#continuous#continuous

- 增加ContinuousField
- 修改_feature_tokenize@text_dataset.py
- 修改text_fields@text_dataset.py
- 修改count_features@preprocess.py
"""

from torchtext.data import RawField
from functools import partial


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


def _control_signal_type(string, layer=0, tok_delim=None, discrete_cs_delim=None, continuous_cs_delim=None, truncate=None):
    """
    获取token_type

        - token_type: token, discrete_control_signal, continuous_contorl_signal
    """

    cs_split = [e for e in string if e in [discrete_cs_delim, continuous_cs_delim]]

    if layer == 0:
        token_type = "token"
    elif cs_split[layer - 1] == continuous_cs_delim:
        token_type = "continuous_control_signal"
    elif cs_split[layer - 1] == discrete_cs_delim:
        token_type = "discrete_control_signal"
    else:
        raise ValueError("bad string")

    return token_type


def _control_signal_tokenize(
        string, layer=0, tok_delim=None, discrete_cs_delim=None, continuous_cs_delim=None, truncate=None):
    """Split apart control signal (like length[continuous] cate_id/author_id[discrete]) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and control signal joined by ``discrete_cs_delim`` and ``continuous_cs_delim``. For example,
            ``"hello|NOUN#8 Earth|NOUN#8"``.
        layer (int): Which feature to extract. (Not used if there are no features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of tokens.

    Returns:
        - List[str] of tokens
    """

    tokens = string.split(tok_delim)

    if truncate is not None:
        tokens = tokens[:truncate]

    cs_split = [e for e in string if e in [discrete_cs_delim, continuous_cs_delim]]
    if cs_split:
        tokens = [ee[layer] for t in tokens for e in t.split(discrete_cs_delim) for ee in e.split(continuous_cs_delim)]

    return tokens


def text_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_control_signal (int): Number of control signal (not counting the tokens, including for both continuous and discrete control signal)
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
    fields_ = []

    discrete_cs_delim = u"￨" if n_feats > 0 else None
    continuous_cs_delim = u"￫" if n_feats > 0 else None

    for i in range(n_feats + 1):
        name = base_name + "_ctrl_signal_" + str(i - 1) if i > 0 else base_name
        tokenize = partial(
            _control_signal_tokenize,
            layer=i,
            truncate=truncate,
            discrete_cs_delim=discrete_cs_delim,
            continuous_cs_delim=continuous_cs_delim,
        )
        use_len = i == 0 and include_lengths
        token_type = _control_signal_type(
            layer=i,
            truncate=truncate,
            discrete_cs_delim=discrete_cs_delim,
            continuous_cs_delim=continuous_cs_delim,
        )
        feat = Field(
            init_token=bos, eos_token=eos,
            pad_token=pad, tokenize=tokenize,
            include_lengths=use_len)
        fields_.append((name, feat))
    assert fields_[0][0] == base_name  # sanity check
    field = TextMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    return field
