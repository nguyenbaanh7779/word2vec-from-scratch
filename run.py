
import typer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
import sys
sys.path.append(str(SRC))

app = typer.Typer(help="Word2Vec from Scratch - Run Experiments")

@app.callback()
def main():
    # with a single command Typer would make it the root command and reject `run.py train ...`;
    # registering a callback keeps `train` as an explicit subcommand
    pass

@app.command()
def train(
    run_name: str = typer.Option(..., help="A name for this training run, used for saving the resulting embeddings"),
    embed_dim: int = typer.Option(100, help="Dimensionality of the word embeddings"),
    epochs: int = typer.Option(1, help="Number of training epochs"),
    max_window: int = typer.Option(5, help="Maximum context window size"),
    n_negatives: int = typer.Option(5, help="Number of negative samples to draw per (centre, context) pair"),
    max_tokens: int = typer.Option(1_000_000, help="Number of tokens from text8 to use for training")
):
    from training.train import train
    train(run_name, embed_dim, epochs, max_window, n_negatives, max_tokens)


if __name__ == "__main__":
    app()
    
