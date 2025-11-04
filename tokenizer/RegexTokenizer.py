from .Tokenizer import Tokenizer
import re


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=r"[\w']+"):
        self.pattern = pattern
        self.regex = re.compile(pattern)

    def tokenize(self, text):
        return self.regex.findall(text)
