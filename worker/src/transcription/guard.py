"""Segment-local lexical repetition guard; leave timestamp tokens untouched."""
import re
from transformers import LogitsProcessor

class SegmentGuard(LogitsProcessor):
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.events = 0
        self.safe = {}
    def __call__(self, input_ids, scores):
        ids = input_ids[0, self.prompt_len:].tolist()
        # Only inspect text after a complete speaker marker. Closing/opening
        # timestamps disable the guard until a new speaker marker is complete.
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        match = re.search(r'\[(?:S\d+|MULTI)\]([^\[\]]*)$', text)
        if not match:
            return scores
        tail = self.tokenizer.encode(match[1], add_special_tokens=False)
        for n in range(1, 7):
            if len(tail) < n*5:
                continue
            prefix = tail[-(n-1):] if n > 1 else []
            counts = {}
            for i in range(len(tail)-n+1):
                gram = tail[i:i+n]
                if gram[:-1] == prefix:
                    counts[gram[-1]] = counts.get(gram[-1], 0)+1
            for token, count in counts.items():
                if count < 4:
                    continue
                if token not in self.safe:
                    decoded = self.tokenizer.decode([token])
                    self.safe[token] = bool(decoded.strip()) and not any(c in decoded for c in '[]0123456789') and token not in self.tokenizer.all_special_ids
                if self.safe[token]:
                    scores[0, token] = -float('inf')
                    self.events += 1
        return scores
