from numpy.random import binomial


class BinomialBandit:
    """
    Implementación de un Bandido Multibrazo (Multi-Armed Bandit) basado
    en una distribución binomial

    Parámetros
    ----------
    number: integer
        Número de recompensas que puede devolver el agente
    probability : float
        Probabilidad de que el objeto devuelva una recompensa
    
    Métodos
    -------
    pull :
        Realiza una tirada en el bandido
        
    """
    def __init__(self, probability, number=1):
        self.number = number
        self.probability = probability
        
        self.reward = self.number * self.probability
        
        
    def pull(self):
        """ Realiza una tirada en el bandido

        Retorna
        -------
        reward: float
            Recompensa obtenida en la tirada
        """  
        return binomial(self.number, self.probability)