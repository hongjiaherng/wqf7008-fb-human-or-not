# WQF7008 Practical Deep Learning - Project

Kaggle's "Facebook Recruiting IV: Human or Robot?"

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

Install the dev dependencies:

```bash
uv sync --dev --extra cpu
# If you have a CUDA-enabled GPU, you can install the GPU version:
# uv sync --dev --extra cu130
```

Add extra packages if needed:

```bash
uv add <package-name>
```

## Development

Use jupyter lab/notebook for development:

```bash
# Do this
.venv/Scripts/activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
jupyter lab # jupyter notebook

# Or this
uv run jupyter lab  # uv run jupyter notebook
```

Or use any IDE/text editor.

## References

### Competition

- Kaggle: [Facebook Recruiting IV: Human or Robot?](https://kaggle.com/competitions/facebook-recruiting-iv-human-or-bot) (2015)

```bibtex
@misc{facebook-recruiting-iv-human-or-bot,
    author = {Jim Dullaghan and John Costella and John_W and Meghan O'Connell and Rafael and Ruchi and RuchiVarshney and Sergey and Sofus Macskassy and Wendy Kan},
    title  = {Facebook Recruiting IV: Human or Robot?},
    year   = {2015},
    howpublished = {\url{https://kaggle.com/competitions/facebook-recruiting-iv-human-or-bot}},
    note   = {Kaggle}
}
```

### Top solutions

Scores reported as ROC-AUC (private / public leaderboard).

| Rank | Score (private / public) | Writeup |
| ---- | ------------------------ | ------- |
| 1st  | 0.94254 / 0.91946        | [Forum comment by the winner](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce#81331) |
| 2nd  | 0.94167 / 0.93277        | [small-yellow-duck: "Share your secret sauce"](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce), [blog post](http://small-yellow-duck.github.io/auction.html) |
| 3rd  | 0.94113 / 0.93321        | [Forum comment by mechatroner](https://www.kaggle.com/competitions/facebook-recruiting-iv-human-or-bot/writeups/small-yellow-duck-share-your-secret-sauce#81396) |
