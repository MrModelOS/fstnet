import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import List


IGNORE = -100


class InstructDataset(Dataset):
    """Instruction датасет (CodeAlpaca): фиксированные seq_len примеры с маской loss."""

    def __init__(self, data_dir: str = "data/instruct", seq_len: int = 256, max_samples: int = 100_000):
        self.seq_len = seq_len
        meta_path = os.path.join(data_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Instruction data not found: {data_dir}")
        meta = json.load(open(meta_path))
        n = min(meta["n"], max_samples)

        self.x = np.memmap(
            os.path.join(data_dir, "train_x.bin"), dtype=np.uint32, mode="r",
            shape=(meta["n"], seq_len),
        )
        self.y = np.memmap(
            os.path.join(data_dir, "train_y.bin"), dtype=np.int32, mode="r",
            shape=(meta["n"], seq_len),
        )
        self.n = n
        print(f"InstructDataset: {n:,} samples, seq={seq_len}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx].astype(np.int64), dtype=torch.long)
        y = torch.tensor(self.y[idx].astype(np.int64), dtype=torch.long)
        return x, y


class LocalTextDataset(Dataset):
    """Датасет из предобработанного бинарного массива токенов (memmap)."""

    def __init__(
        self,
        data_dir: str = "data/local",
        tokenizer_path: str = "tokenizer/fst_bpe.json",
        seq_len: int = 256,
        max_samples: int = 100_000,
    ):
        self.seq_len = seq_len
        self.max_samples = max_samples

        self.tokenizer_path = tokenizer_path
        bin_path = os.path.join(data_dir, "train.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Preprocessed tokens not found: {bin_path}")

        # memmap: не держим всё в RAM, читаем страницами по индексу
        n_tokens = os.path.getsize(bin_path) // 4
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(n_tokens,))
        print(f"LocalTextDataset: {n_tokens:,} tokens (memmap, {os.path.getsize(bin_path)/1e6:.0f}MB)")

        self.n_samples = min(
            self.max_samples, max(0, (n_tokens - 1) // seq_len)
        )
        print(f"LocalTextDataset: {self.n_samples:,} samples, seq={seq_len}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        x = torch.tensor(chunk[:-1].astype(np.int64), dtype=torch.long)
        y = torch.tensor(chunk[1:].astype(np.int64), dtype=torch.long)
        return x, y


class SyntheticTextDataset(Dataset):
    """Синтетический датасет на BPE-токенизаторе (fallback)."""

    def __init__(
        self,
        tokenizer_path: str = "tokenizer/fst_bpe.json",
        seq_len: int = 256,
        max_samples: int = 50_000,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.samples = self._generate(seed)

    def _generate(self, seed: int) -> List[torch.Tensor]:
        import random as pyrandom

        rng = pyrandom.Random(seed)

        templates = [
            "The {noun} {verb} through the {adj} {place}.",
            "When {name} arrived, everyone was {emotion}.",
            "To understand {concept}, one must first study {topic}.",
            "The relationship between {noun} and {noun2} is complex.",
            "In the beginning, there was only {noun}.",
            "She looked at the {noun} and felt {emotion}.",
            "The old {person} told a story about {topic}.",
            "If you want to {adv} {verb}, you must practice daily.",
            "Knowledge of {topic} requires patience and dedication.",
            "The {adj} {noun} stood silently in the {place}.",
            "Programming is the art of telling a computer what to do.",
            "Neural networks learn patterns from data through optimization.",
            "Language models predict the next token in a sequence.",
            "The fractal structure allows recursive thinking in latent space.",
            "Memory systems store facts outside the model weights.",
        ]

        nouns = [
            "tree", "river", "mountain", "book", "computer", "idea",
            "algorithm", "neuron", "network", "story", "city", "mind",
            "program", "language", "memory", "model", "system",
        ]
        verbs = [
            "runs", "flows", "grows", "develops", "evolves", "connects",
            "learns", "thinks", "creates", "transforms", "processes",
        ]
        adj = [
            "beautiful", "complex", "simple", "deep", "bright", "dark",
            "ancient", "modern", "fractal", "recursive", "elegant",
        ]
        places = [
            "forest", "city", "ocean", "desert", "laboratory",
            "universe", "mind", "network", "datacenter",
        ]
        concepts = [
            "intelligence", "learning", "memory", "language", "logic",
            "consciousness", "computation", "reasoning",
        ]
        emotions = [
            "happy", "curious", "thoughtful", "amazed", "concerned",
            "hopeful", "inspired",
        ]
        names = [
            "Alex", "Maria", "Leo", "Sophia", "Ivan", "Anna", "Max", "Elena",
        ]
        topics = [
            "mathematics", "philosophy", "physics", "computer science",
            "neuroscience", "linguistics", "engineering",
        ]
        adverbs = ["quickly", "carefully", "deeply", "precisely", "efficiently"]
        people = ["man", "woman", "teacher", "scientist", "writer", "thinker"]

        samples = []
        for _ in range(self.max_samples):
            seq_tokens = []
            while len(seq_tokens) < self.seq_len + 1:
                template = rng.choice(templates)
                try:
                    text = template.format(
                        noun=rng.choice(nouns),
                        noun2=rng.choice(nouns),
                        verb=rng.choice(verbs),
                        adj=rng.choice(adj),
                        place=rng.choice(places),
                        concept=rng.choice(concepts),
                        emotion=rng.choice(emotions),
                        name=rng.choice(names),
                        topic=rng.choice(topics),
                        adv=rng.choice(adverbs),
                        person=rng.choice(people),
                    )
                except KeyError:
                    text = template

                tokens = self.tokenizer.encode(" " + text).ids
                seq_tokens.extend(tokens)

            seq_tokens = seq_tokens[: self.seq_len + 1]
            samples.append(torch.tensor(seq_tokens, dtype=torch.long))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return chunk[:-1], chunk[1:]
