import mablane.bandits

from mablane.bandits import BinomialBandit

def test_binomial_bandit():
    mablane.bandits._BinomialBandit.binomial = lambda n, p: n + p

    bandit = BinomialBandit(1)

    assert bandit.pull() == 2

    bandit = BinomialBandit(1, 2)

    assert bandit.pull() == 3