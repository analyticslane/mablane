import mablane

from mablane.bandits import NegativeBinomialBandit

def test_binomial_bandit():
    mablane.bandits._NegativeBinomialBandit.negative_binomial = lambda n, p: n + p

    bandit = NegativeBinomialBandit(1)

    assert bandit.pull() == 2

    bandit = NegativeBinomialBandit(1, 2)

    assert bandit.pull() == 3