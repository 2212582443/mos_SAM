import torch
from torch.nn import GRU
from torch.nn import Module, Linear, ReLU, Linear, ModuleList, functional as F

from .typing import SparseEmbeddingsTensor, TextTokenEmbeddingTensor

from ...configuration_sam import SamPromptEncoderConfig
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class _TextTensorlizer:
    def __init__(self):
        self._tokenizer = None
        self._model = None

    def _lazy_init(self):
        local_files_only = True
        self._tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased", local_files_only=local_files_only
        )
        self._model = BertModel.from_pretrained("bert-base-multilingual-cased", local_files_only=local_files_only)
        # text = "Replace me by any text you'd like."
        # encoded_input = tokenizer(text, return_tensors='pt')
        # tokenizer.decode(encoded_input['input_ids'][0]) -> "[CLS] Replace me by any text you'd like. [SEP]"
        # output = model(**encoded_input)

    def text2tensor(self, text: str, max_length=40) -> TextTokenEmbeddingTensor:
        """
        return:
            (1, seq_len, text_hidden_size)
        """
        if self._tokenizer is None:
            self._lazy_init()
        result: BaseModelOutputWithPoolingAndCrossAttentions = self._model(
            **self._tokenizer(text, padding="max_length", max_length=max_length, return_tensors="pt")
        )
        return result.last_hidden_state


_text_tensorlizer = _TextTensorlizer()


def text2tensor(text: str, max_length=40) -> TextTokenEmbeddingTensor:
    """
    return:
        (1, seq_len, text_hidden_size)
    """
    return _text_tensorlizer.text2tensor(text, max_length)


# todo, 改用其他方法提取句子的embedding, e.g.
#   On the Sentence Embeddings from Pre-trained Language Models
#       https://arxiv.org/pdf/2011.05864.pdf
#       https://github.com/bohanli/BERT-flow


class TextEmbeddingMLP(Module):
    def __init__(
        self,
        config: SamPromptEncoderConfig,
        input_dim: int = 768,
        num_layers: int = 3,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = ReLU()
        self.proj_in = Linear(input_dim, config.hidden_size)
        self.layers = ModuleList([Linear(config.hidden_size, config.hidden_size) for _ in range(num_layers - 2)])
        self.proj_out = Linear(config.hidden_size, config.hidden_size)
        self.sigmoid_output = sigmoid_output

    def forward(self, token_embedding: TextTokenEmbeddingTensor) -> SparseEmbeddingsTensor:
        """
        Args:
            token_embedding: (bs, seq_len, token_hidden_size)

        Returns:
            token_embedding: (bs, 1, hidden_size)
        """

        token_embedding = token_embedding.to(self.proj_in.weight.device)
        token_embedding = token_embedding.sum(dim=1)
        token_embedding = self.proj_in(token_embedding)
        token_embedding = self.activation(token_embedding)
        for layer in self.layers:
            token_embedding = self.activation(layer(token_embedding))

        token_embedding = self.proj_out(token_embedding)
        if self.sigmoid_output:
            token_embedding = F.sigmoid(token_embedding)
        token_embedding = token_embedding.unsqueeze(1)
        return token_embedding


class TextEmbeddingGRU(Module):
    def __init__(
        self,
        config: SamPromptEncoderConfig,
        input_dim: int = 768,
        num_layers: int = 3,
    ):
        super().__init__()
        self.gru = GRU(
            input_dim,
            config.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.h0 = torch.nn.Parameter(
            torch.randn(num_layers * 2, 1, config.hidden_size),
            requires_grad=True,
        )
        self.proj_out = Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, token_embedding: TextTokenEmbeddingTensor) -> SparseEmbeddingsTensor:
        """
        Args:
            token_embedding: (bs, seq_len, token_hidden_size)

        Returns:
            token_embedding: (bs, 1, hidden_size)
        """
        bs = token_embedding.shape[0]
        h0 = self.h0.repeat(1, bs, 1)
        # output (bs, seq_len, 2 * hidden_size)
        # h_n: (num_layers * num_directions, bs, hidden_size)
        output, h_n = self.gru(token_embedding, h0)
        token_embedding = output[:, -1:, :]
        token_embedding = self.proj_out(token_embedding)
        return token_embedding
