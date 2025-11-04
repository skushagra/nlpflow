from .Tokenizer import Tokenizer


class SplitTokenizer(Tokenizer):
    def __init__(self, delimiter=" "):
        self.delimiter = delimiter

    def tokenize(self, text):
        return text.split(self.delimiter)
