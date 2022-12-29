import datetime
import os

import pandas as pd
import numpy as np

import xgboost as xgb


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def process_schedule_data(schedule_data: pd.DataFrame):
    """Process schedule reliability data

    Args:
        schedule_data (pd.DataFrame): raw data read from reliability schedule

    Returns:
        pd.DataFrame: processed dataframe excluding rows with null labels,
        creating new columns 'Date' and 'Avg_TurnoverDays'
    """
    # process schedule data
    # exclude rows with null reliability values
    rel_df_nona = schedule_data[~schedule_data["OnTime_Reliability"].isna()]

    # add date column
    # convert 3-letter month abbrev to integer equivalent
    rel_df_nona["Month(int)"] = rel_df_nona[
        "Month"
    ].apply(
        lambda x:
        datetime.datetime.strptime(x, '%b').month
    )
    # add date
    rel_df_nona["Date"] = rel_df_nona.apply(
        lambda x: datetime.datetime(
            x["Calendary_Year"], x["Month(int)"], 1
        ), axis=1
    )

    # change target field data type to float
    rel_df_nona.loc[:, "OnTime_Reliability"] = rel_df_nona[
        "OnTime_Reliability"
    ].apply(lambda x: float(x[:-1]))

    # create new variable
    # Avg_TurnoverDays = Avg_TTDays + Avg_WaitTime_POD_Days
    rel_df_nona.loc[:, "Avg_TurnoverDays"] = rel_df_nona[
        "Avg_TTDays"
    ] + rel_df_nona["Avg_WaitTime_POD_Days"]

    return rel_df_nona


def restrict_by_coverage(rel_df_nona: pd.DataFrame, min_no_months=9):
    """Restrict to carrier service routes with given no. of months covered

    Args:
        rel_df_nona (pd.DataFrame): shipping schedule dataframe
        min_no_months (int, optional): months threshold. Defaults to 9.

    Returns:
        pd.DataFrame: dataframe with routes having at least nine months' worth of data
    """
    rel_df_nona_cvg = rel_df_nona.groupby(
        ["POL", "POD", "Carrier", "Service"]
    ).apply(lambda x: len(x["Month"].unique())
    )

    rel_df_nona_full_cvg = rel_df_nona_cvg[rel_df_nona_cvg==min_no_months]

    rel_df_nona_full_cvg_indices = rel_df_nona_full_cvg.index

    base_features = zip(
        rel_df_nona["POL"],
        rel_df_nona["POD"],
        rel_df_nona["Carrier"],
        rel_df_nona["Service"]
    )

    new_indices = []
    for idx, base_feature in enumerate(base_features):
        if base_feature in rel_df_nona_full_cvg_indices:
            new_indices.append(idx)


    return rel_df_nona.iloc[new_indices, :]


def split_data(
    rel_df_nona: pd.DataFrame,
    datetime_split: datetime.datetime,
    max_month=9
):
    """Return train, val split for baseline model

    Args:
        rel_df_nona (pd.DataFrame): shipping schedule data
        datetime_split (datetime.datetime): time horizon
        max_month (int, optional): no. of months to cover. Defaults to 9.

    Returns:
        tuple: (unique 'POD, POL, Carrier, Service' rows with weighted avg and std,
               filtered validation data set with common 'POD, POL, Carrier, Service'
               values from both train and val)
    """

    month_thresh = datetime.datetime(2022, max_month, 1)
    # train
    train = rel_df_nona[rel_df_nona["Date"] < datetime_split]

    # val
    val = rel_df_nona[
        (rel_df_nona["Date"] >= datetime_split) &
        (rel_df_nona["Date"] <= month_thresh)
    ]


    # let's get multi-index pairs from train
    train_indices = list(
        train[
            ["Carrier", "Service", "POD", "POL"]
        ].groupby(["Carrier", "Service", "POD", "POL"]).count().index
    )

    # now find the intersection between train and val
    indices_inter = []
    for ind, row in val.iterrows():
        ind_pair = (row["Carrier"], row["Service"], row["POD"], row["POL"])
        if ind_pair in train_indices:
            indices_inter.append(ind)

    # now restrict to the indices in the intersection
    val_res = val.loc[indices_inter, :]

    return train, val_res


def load_excel_data(config: dict, data_name: str):
    """Load excel data corresp. to data name

    Args:
        config (dict): config dict consisting of data and eval params
        data_name (str): string representing data name (e.g. port call or retail sales)

    Returns:
        pd.DataFrame: dataframe corresponding to data_name
    """

    filename = config[data_name]["filename"]
    sheetname = config[data_name]["sheet"]

    data_dir = config["data_path"]

    path = os.path.join(
        data_dir,
        filename
    )
    data = pd.read_excel(
        path,
        sheet_name=sheetname
    )

    return data


def weighted_average_ser(ser: pd.Series):
    """Return weighted average of series

    Args:
        ser (pd.Series): pandas Series with float values

    Returns:
        float: weighted average of series
    """

    wts = pd.Series([1 / val if val != 0 else 0 for val in ser])

    if wts.sum() == 0: return 0  # avoid division by zero

    return (ser * wts).sum() / wts.sum()


def get_carr_serv_mask(df: pd.DataFrame, carrier: str, service: str):
    """Return pandas series mask given specific carrier and service

    Args:
        df (pd.DataFrame): Dataframe containing columns 'Carrier' and 'Service'
        carrier (str): string corresp. to a specific carrier
        service (str): string corresp. to a specific service

    Returns:
        pd.Series: series mask corresponding to carrier and service
    """
    return (df["Carrier"]==carrier) & \
        (df["Service"]==service)


def get_reg_train_test(
    df: pd.DataFrame,
    datetime_split: datetime.datetime,
    label='Avg_TTDays',
    use_retail=False
):
    """Return train val data with train weighted mean features

    Args:
        df (pd.DataFrame): shipping + features train data
        datetime_split (datetime.datetime): time horizon to split from
        label (str, optional): column label. Defaults to 'Avg_TTDays'.
        use_retail (bool, optional): whether retail features are included. Defaults to False.

    Returns:
        tuple: train, val split
    """

    date_column = "Date"
    # train
    train = df[df[date_column] < datetime_split]

    train_wt_mean = get_train_wt_avg(train, datetime_split, label=label)

    train_wt_mean.columns = [
        'Carrier',
        'Service',
        'POD',
        'POL',
        f'{label}_train',
        f'{label}(std)_train'
    ]

    train_min = get_train_wt_avg(train, datetime_split, label=label, agg_fun=np.min)
    train_min.columns = [
        'Carrier',
        'Service',
        'POD',
        'POL',
        f'{label}_min_train',
        f'{label}(std)_min_train'
    ]


    train_max = get_train_wt_avg(train, datetime_split, label=label, agg_fun=np.max)
    train_max.columns = [
        'Carrier',
        'Service',
        'POD',
        'POL',
        f'{label}_max_train',
        f'{label}(std)_max_train'
    ]


    train = train_wt_mean.merge(train, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    train = train_min.merge(train, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    train = train_max.merge(train, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    # val
    val = df[df[date_column] >= datetime_split]

    val = train_wt_mean.merge(val, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    val = train_min.merge(val, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    val = train_max.merge(val, on=[
        'Carrier',
        'Service',
        'POD',
        'POL'
    ])

    # predictors
    if not use_retail:
        predictors = [
            "POL",
            "POD",
            "Carrier",
            "Service",
            "Trade",
            "Avg_Port_Hours(by_call)",
            "Avg_Anchorage_Hours(by_call)",
            f"{label}_train",
            f"{label}(std)_train",
            f"{label}_min_train",
            f"{label}_max_train"
        ]
    else:

        train.rename(columns={"retail_sales_x": "retail_sales"}, inplace=True)
        val.rename(columns={"retail_sales_x": "retail_sales"}, inplace=True)

        # drop retail sales na values
        train = train[~train["retail_sales"].isna()]
        val = val[~val["retail_sales"].isna()]

        predictors = [
            "POL",
            "POD",
            "Carrier",
            "Service",
            "Trade",
            "retail_sales",
            f"{label}_train",
            f"{label}(std)_train",
            f"{label}_min_train",
            f"{label}_max_train",
        ]


    train_X, train_y = train[predictors], train[label]
    val_X, val_y = val[predictors], val[label]

    return train_X, train_y, val_X, val_y


def get_train_wt_avg(
    rel_df_nona: pd.DataFrame,
    datetime_split: datetime.datetime,
    label="Avg_TTDays",
    agg_fun=weighted_average_ser
):
    """_summary_

    Args:
        rel_df_nona (pd.DataFrame): shipping schedule data + features
        datetime_split (datetime.datetime): time horizon
        label (str, optional): target label. Defaults to "Avg_TTDays".
        agg_fun (Callable, optional): aggregate function to weigh values by. Defaults to weighted_average_ser.

    Returns:
        pd.DataFrame: dataframe with unique "Carrier", "Service", "POL", "POD" rows
                    and label weighted by aggregate function
    """

    train = rel_df_nona[rel_df_nona["Date"] < datetime_split]

    # weighted average
    train_on_time_rel_by_carr_ser = train[[
        "Carrier", "Service", "POD", "POL", label,
    ]].groupby(["Carrier", "Service", "POD", "POL"]).apply(lambda x: (agg_fun(x[label].values), x[label].values.std())).reset_index()

    train_on_time_rel_by_carr_ser.loc[:, f"{label}"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[0])
    train_on_time_rel_by_carr_ser.loc[:, f"{label}(std)"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[1])

    train_on_time_rel_by_carr_ser.drop(0, axis=1, inplace=True)

    train_df = train_on_time_rel_by_carr_ser.copy()

    return train_df


def gen_lag(
    df: pd.DataFrame,
    lag=1,
    lag_col="Month(int)",
    target_cols=["OnTime_Reliability"],
    common_cols=["Carrier", "Service", "POL", "POD", "Trade", "Month(int)"]
):
    """Generate month lag on a target column by some number of months.

    Args:
        df (pd.DataFrame): train data
        lag (int, optional): no. of months to lag by. Defaults to 1.
        lag_col (str, optional): lag column name. Defaults to "Month(int)".
        target_cols (list, optional): target column list. Defaults to ["OnTime_Reliability"].
        common_cols (list, optional): columns to merge on.
            Defaults to ["Carrier", "Service", "POL", "POD", "Trade", "Month(int)"].

    Returns:
        pd.DataFrame: dataframe with new lag column appended
    """

    # make a copy of dataframe
    # to apply lag
    df_lag = df.copy()

    # apply lag
    df_lag.loc[:, lag_col] += 1

    # rename the target columns

    for target_col in target_cols:
        new_col_name = f"{target_col}_lag_{lag}"
        df_lag.loc[:, new_col_name] = df_lag[target_col]
        df_lag.drop(target_col, inplace=True, axis=1)

    # now merge lagged feature onto original df
    df_with_lag_feature = df.merge(df_lag, on=common_cols)
    df_with_lag_feature.rename(

        columns={
            "Date_x": "Date",
            "Avg_TTDays_x": "Avg_TTDays",
            "Avg_WaitTime_POD_Days_x": "Avg_WaitTime_POD_Days",
            "Avg_Port_Hours(by_call)_x": "Avg_Port_Hours(by_call)",
            "Avg_Anchorage_Hours(by_call)_x": "Avg_Anchorage_Hours(by_call)"
        },
        inplace=True
    )

    return df_with_lag_feature


def filter_nonzero_values(data_X, data_y, preds, label):
    """Filter feature and target data with nonzero labels

    Args:
        data_X (pd.DataFrame): feature data
        data_y (pd.Series): label data
        preds (list): model predictions
        label (str): target label

    Returns:
        tuple: filtered feature, labels, and predictions
    """
    preds_array = np.array(preds)

    # create mask to use for filtering
    nonzero_mask = data_y != 0
    nonzero_mask = nonzero_mask.reset_index()[label]

    # filtering zero values
    if sum(nonzero_mask) != 0:

        preds = pd.Series(preds)[nonzero_mask]

        data_y = data_y.reset_index()[label]
        data_y = data_y[nonzero_mask]

        data_X = data_X.reset_index().drop("index", axis=1)
        data_X = data_X[nonzero_mask]

        preds_array = np.array(preds)

        data_gt = data_y.values

        return data_X, data_gt, preds_array

    else:
        raise Exception("All target values are zero!")


def compute_eval_metrics(
    model,
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    train_y: pd.Series,
    val_y: pd.Series,
    include_overestimates=False,
    label="Avg_TTDays"
):
    """Return train/validation MAE and MAPE metrics given scikit learn model

    Args:
        model (sklearn.base.BaseEstimator): scikit estimator object
        train_X (pd.DataFrame): feature data
        val_X (pd.DataFrame): val data
        train_y (pd.Series): train ground truth
        val_y (pd.Series): val ground truth
        include_overestimates (bool, optional): whether to calculate MAPE. Defaults to False.
        label (str, optional): target label. Defaults to "Avg_TTDays".

    Returns:
        tuple: MAE and MAPE values for validation
    """

    train_X_rg, val_X_rg = compute_common_cols(train_X, val_X)

    # fit model
    model.fit(train_X_rg, train_y)

    train_preds = model.predict(train_X_rg)
    val_preds = model.predict(val_X_rg)

    # need to make sure reliability predictions are capped at 100 and 0
    if label == "Avg_TTDays":
        train_preds = list(map(lambda x: 100 if x >= 100 else x, model.predict(train_X_rg)))
        val_preds = list(map(lambda x: 100 if x >= 100 else x, model.predict(val_X_rg)))

        train_preds = list(map(lambda x: 0 if x<=0 else x, train_preds))
        val_preds = list(map(lambda x: 0 if x<=0 else x, val_preds))

    # val metrics
    val_X, val_gt, preds_array = filter_nonzero_values(val_X, val_y, val_preds, label)
    val_mae = mean_absolute_error(val_gt, preds_array)
    val_mape = mean_absolute_percentage_error(val_gt, preds_array)

    # train metrics
    train_X, train_gt, train_preds_array = filter_nonzero_values(train_X, train_y, train_preds, label)
    train_mae = mean_absolute_error(train_gt, train_preds_array)
    train_mape = mean_absolute_percentage_error(train_gt, train_preds_array)

    # create error result dataframe
    result_df = val_X.copy()
    result_df.loc[:, "actual"] = val_y
    result_df.loc[:, "pred"] = preds_array
    result_df.loc[:, "error"] = preds_array - val_y
    result_df.loc[:, "perc_error"] = (preds_array - val_y) / val_y

    result_df = result_df[
        [
            "Carrier",
            "Service",
            "POD",
            "POL",
            "actual",
            "pred",
            "error",
            "perc_error"
        ]
    ]

    # overestimate mask
    diff = preds_array - val_y
    mask = diff > 0

    # mape
    if include_overestimates:

        if sum(mask) == 0: raise Exception("There are not overestimated preds!")

        # compute mae (over)
        val_mae_over = diff[mask].mean()
        # series mask
        mask_ser = mask.reset_index()[label]
        # compute mape (over)
        val_preds_over = pd.Series(preds_array)[mask_ser]
        val_y_over = pd.Series(list(val_y))[mask_ser]
        val_mape_over = mean_absolute_percentage_error(val_y_over, val_preds_over)


        return train_mae, train_mape, val_mae, val_mape, val_mae_over, val_mape_over, result_df

    return train_mae, train_mape, val_mae, val_mape, result_df


# we need to restrict to common inputs
def compute_common_cols(train_X: pd.DataFrame, val_X: pd.Series):
    """Return train and val restricted to common cols

    Args:
        train_X (pd.DataFrame): feature data
        val_X (pd.Series): test data

    Returns:
        tuple: feature and test data restricted to common columns (for regression fitting)
    """

    # get dummies for categorical variables
    train_X_rg = pd.get_dummies(train_X)
    val_X_rg = pd.get_dummies(val_X)

    # restrict to common columns
    common_cols = list(
        set(train_X_rg.columns).intersection(
            set(val_X_rg.columns)
        )
    )

    return train_X_rg[common_cols], val_X_rg[common_cols]
