class Vocab:
    def __init__(self, vocab):
        self.tokens = vocab
        self.tokens_to_ids = {token: id_ for id_, token in enumerate(vocab)}
        self.ids_to_tokens = {id_: token for id_, token in enumerate(vocab)}
        self.size = len(self.tokens)

    def get_vocabulary(self, level: int = 0):
        """Just using this for diagnosis."""
        if level == 0:
            return self.tokens
        elif level == 1:
            return self.tokens_to_ids
        elif level == 2:
            return self.ids_to_tokens
        return self.tokens

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.ids_to_tokens
        if isinstance(item, str):
            return item in self.tokens_to_ids
        raise TypeError("Item must be an int or a str")

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.ids_to_tokens[item]
        if isinstance(item, str):
            return self.tokens_to_ids[item]
        raise TypeError("Item must be an int or a str")

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.tokens_to_ids)
