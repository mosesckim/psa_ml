import pandas as pd

from ontime.src.utils import weighted_average_ser


class BaselineModel:
    """
    A class the weightedd baseline model


    ...

    Attributes
    ----------
    train_df : pd.DataFrame
        pandas data frame containing unique (POD, POL, Carrier, Service) tuples
    label : str
        target label (to predict)


    Methods
    -------
    predict(carrier, service, pod, pol)
        Returns the label weighted value  and standard deviation
        corresp. to given carrier, service, pod, and pol.

    """

    def __init__(self):
        """Baseline model class constructor
        """
        self.train_df = None
        self.label = None

    def fit(self, train: pd.DataFrame, label="OnTime_Reliability"):
        """Fit baseline model to train data

        Args:
            train (pd.DataFrame): train dataframe containing columns "Carrier", "Service", "POD", and "POL".
            label (str, optional): Target label. Defaults to "OnTime_Reliability".
        """
        # TODO: perform column existence check
        train_on_time_rel_by_carr_ser = train[[
            "Carrier", "Service", "POD", "POL", label
        ]].groupby(["Carrier", "Service", "POD", "POL"]).apply(
            lambda x: (weighted_average_ser(x[label].values), x[label].values.std())
        ).reset_index()

        train_on_time_rel_by_carr_ser.loc[:, f"{label}"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[0])
        train_on_time_rel_by_carr_ser.loc[:, f"{label}(std)"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[1])

        train_on_time_rel_by_carr_ser.drop(0, axis=1, inplace=True)

        self.label = label
        self.train_df = train_on_time_rel_by_carr_ser

    def predict_(self, carrier: str, service: str, pod: str, pol: str):
        """Return weighted label value and standard deviation

        Args:
            carrier (str): carrier or shipping (e.g. MAERSK)
            service (str): service string or route designation
            pod (str): port of destination
            pol (str): port of loading

        Returns:
            tuple: weighted label value, standard deviation
        """

        # apply mask
        pred = self.train_df[
            (self.train_df["Carrier"]==carrier) & (self.train_df["Service"]==service) & \
            (self.train_df["POD"]==pod) & (self.train_df["POL"]==pol)
        ]

        # predict label
        label_pred = pred[self.label]
        # predict interval
        label_pred_std = pred[f"{self.label}(std)"]

        return label_pred.iloc[0], label_pred_std.iloc[0]

    def predict(self, test_data):
        """Return weighted label value and standard deviation

        Args:
            test_data (pd.DataFrame): test dataframe containing cols "Carrier", "Service", "POD", "POL"

        Returns:
            tuple: weighted label value, standard deviation
        """

        # merge train data
        preds = test_data.merge(
            self.train_df,
            on=["Carrier", "Service", "POD", "POL"]
        )

        label_preds = preds[self.label]
        label_preds_std = preds[f"{self.label}(std)"]

        return label_preds, label_preds_std
