# scripts containing

# moving average (baseline model)
# linear regressor

# xgboost models for transit time prediction


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

    def __init__(self, train_df, label="OnTime_Reliability"):
        """_summary_

        Args:
            train_df (DataFrame): _description_
            label (str, optional): _description_. Defaults to "OnTime_Reliability".
        """
        self.train_df = train_df
        self.label = label

    def predict(self, carrier: str, service: str, pod: str, pol: str):
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
