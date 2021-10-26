"""Generate sentiment analysis data."""
import numpy as np

VOCAB = ["good", "bad", "uh"]

VOCAB2 = ["one", "two", "three", "four", "five"]


def gen_gbu(nobs=1000):
    """Generate good/bad/uh data."""
    data = []
    sentiments = []
    for _ in range(nobs):
        num_positions = np.random.randint(5, 20)
        words = np.random.choice(
            VOCAB,
            num_positions,
            p=np.random.dirichlet([0.2, 0.2, 0.2])
        )
        sentiments.append(np.sum(words == "good") - np.sum(words == "bad"))
        data.append(list(words))
    return data, sentiments


def main():
    """Test GBU data."""
    X, Y = gen_gbu(10)
    print(X)
    print(Y)


if __name__ == "__main__":
    main()
