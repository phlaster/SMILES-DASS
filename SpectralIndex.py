import re


class SpectralIndex:
    def __init__(self, formula):
        self.formula = formula
        self.available_sensors = {
            "B": 0, "G": 1, "R": 2, "RE1": 3, "RE2": 4, "RE3": 5, "N": 6, "N2": 7, "S1": 8, "S2": 9
        }
        tokens = re.findall(r'\b[A-Za-z]+\b', formula)

        unknown_tokens = [token for token in tokens if token not in self.available_sensors.keys()]
        if unknown_tokens:
            raise ValueError(f"Unknown tokens in formula: {unknown_tokens}")

    def apply(self, array):
        band_vars = {key: array[val] for key, val in self.available_sensors.items()}
        try:
            index_result = eval(self.formula, {}, band_vars)
        except Exception as e:
            raise ValueError(f"Error in evaluating the formula: {e}")

        return index_result