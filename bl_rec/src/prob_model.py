import pandas as pd


class FFProbSpace:
    """Probability space conditioned on filled-in fields
    """

    def __init__(self, df: pd.DataFrame, ff_cols=[]):
        """Constructor

        Args:
            df (pd.DataFrame): BL dataframe
            ff_cols (list, optional): conditional column variables. Defaults to [].
        """

        self.df = df
        self.ff_groups = df.groupby(ff_cols)
        self.prob_sp = None

    def compute_prob_sp(self, uf_col=""):
        """Compute probability space for target column

        Args:
            uf_col (str, optional): target column variable. Defaults to "".
        """

        self.prob_sp = self.ff_groups.apply(
            lambda x: dict(x[uf_col].value_counts())
        )

    def compute_cond_prob(self, ff_values=()):
        """Compute probability values conditioned on a given set of filled-in values

        Args:
            ff_values (tuple, optional): tuple of filled-in values. Defaults to ().

        Returns:
            pd.Series: series consisting of target values and their proabilities
        """

        freq_dict = self.prob_sp[ff_values]
        freq_ser = pd.Series(freq_dict)

        prob_ser = freq_ser / freq_ser.sum()
        return prob_ser.rename(ff_values)
