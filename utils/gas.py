import utils.constants as C

class Gas:
    def __init__(self, name, M, gamma_X):
        self.name = name
        self.M = M
        self.gamma_X = gamma_X



G = dict()

G["CO2"] = Gas("CO2", 44E-3, 1)
G["CH4"] = Gas("CH4", 28E-3, 1.291)
