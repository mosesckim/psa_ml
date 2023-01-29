import datetime
import os

import pandas as pd
import numpy as np

import xgboost as xgb


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

pd.set_option("mode.chained_assignment", None)


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
    rel_df_nona["Month(int)"] = rel_df_nona["Month"].apply(
        lambda x: datetime.datetime.strptime(x, "%b").month
    )
    # add date
    rel_df_nona["Date"] = rel_df_nona.apply(
        lambda x: datetime.datetime(x["Calendary_Year"], x["Month(int)"], 1), axis=1
    )

    # change target field data type to float
    rel_df_nona.loc[:, "OnTime_Reliability"] = rel_df_nona["OnTime_Reliability"].apply(
        lambda x: float(x[:-1])
    )

    # create new variable
    # Avg_TurnoverDays = Avg_TTDays + Avg_WaitTime_POD_Days
    rel_df_nona.loc[:, "Avg_TurnoverDays"] = (
        rel_df_nona["Avg_TTDays"] + rel_df_nona["Avg_WaitTime_POD_Days"]
    )

    return rel_df_nona


def restrict_by_coverage(rel_df_nona: pd.DataFrame, min_no_months=9):
    """Restrict to carrier service routes with given no. of months covered

    Args:
        rel_df_nona (pd.DataFrame): shipping schedule dataframe
        min_no_months (int, optional): months threshold. Defaults to 9.

    Returns:
        pd.DataFrame: dataframe with routes having at least nine months' worth of data
    """
    rel_df_nona_cvg = rel_df_nona.groupby(["POL", "POD", "Carrier", "Service"]).apply(
        lambda x: len(x["Month"].unique())
    )

    rel_df_nona_full_cvg = rel_df_nona_cvg[rel_df_nona_cvg == min_no_months]

    rel_df_nona_full_cvg_indices = rel_df_nona_full_cvg.index

    base_features = zip(
        rel_df_nona["POL"],
        rel_df_nona["POD"],
        rel_df_nona["Carrier"],
        rel_df_nona["Service"],
    )

    new_indices = []
    for idx, base_feature in enumerate(base_features):
        if base_feature in rel_df_nona_full_cvg_indices:
            new_indices.append(idx)

    return rel_df_nona.iloc[new_indices, :]


def split_data(
    rel_df_nona: pd.DataFrame, datetime_split: datetime.datetime, max_month=9
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
        (rel_df_nona["Date"] >= datetime_split) & (rel_df_nona["Date"] <= month_thresh)
    ]

    # let's get multi-index pairs from train
    train_indices = list(
        train[["Carrier", "Service", "POD", "POL"]]
        .groupby(["Carrier", "Service", "POD", "POL"])
        .count()
        .index
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


def align_port_call(
    port_data,
    rel_df_nona,
    agg_cols=["seaport_code", "Month", "Year"],
    target_cols=["Total_Calls", "Port_Hours", "Anchorage_Hours"],
):
    """Compute aggregate by port, month, year, and merge with schedule data

    Args:
        port_data (pd.DataFrame): port call data frame
        rel_df_nona (pd.DataFrame): reliability schedule data frame
        agg_cols (list, optional): aggregate column group. Defaults to ["seaport_code", "Month", "Year"].
        target_cols (list, optional): target colums. Defaults to ["Total_Calls", "Port_Hours", "Anchorage_Hours"].

    Returns:
        pd.DataFrame: port call aggregate data merged with schedule
    """

    # seaport code dict (schedule -> port call)
    seaport_code_map = {"CNSHG": "CNSHA", "CNTNJ": "CNTXG", "CNQIN": "CNTAO"}
    # add seaport_code column to port data
    port_call_df = port_data
    port_call_df.loc[:, "seaport_code"] = port_call_df["UNLOCODE"].apply(
        lambda x: seaport_code_map[x] if x in seaport_code_map else x
    )
    # exclude rows with port code USORF from rel_df since it's missing
    rel_df_no_orf = rel_df_nona[~rel_df_nona.POD.isin(["USORF"])]
    # add seaport code column
    rel_df_no_orf.loc[:, "seaport_code"] = rel_df_no_orf["POD"]
    # compute average hours per call
    # sum up calls, port/anchorage hours
    # and aggregate by port, month, and year
    port_hours_avg = (
        port_call_df[target_cols + agg_cols].groupby(agg_cols).sum().reset_index()
    )
    # average port hours by port, month
    port_hours_avg.loc[:, "Avg_Port_Hours(by_call)"] = (
        port_hours_avg["Port_Hours"] / port_hours_avg["Total_Calls"]
    )
    # average anchorage hours by port, month
    port_hours_avg.loc[:, "Avg_Anchorage_Hours(by_call)"] = (
        port_hours_avg["Anchorage_Hours"] / port_hours_avg["Total_Calls"]
    )
    # merge avg hours
    rel_df_no_orf_pt_hrs = rel_df_no_orf.merge(
        port_hours_avg,
        left_on=["Calendary_Year", "Month(int)", "seaport_code"],
        right_on=["Year", "Month", "seaport_code"],
    )

    return rel_df_no_orf_pt_hrs


def process_sales(sales_data, rel_df_nona):
    """Add features to sales data

    Args:
        sales_data (pd.DataFrame): retail sales data frame
        rel_df_nona (pd.DataFrame): reliability schedule data frame
    """

    # reliability POL mapping -> retail_sales country/region
    rel_port_map = {
        "AEAUH": "Agg Middle East & Africa",
        "AEJEA": "Agg Middle East & Africa",
        "BEANR": "Belgium",
        "BRRIG": "Brazil",
        "CNNGB": "China",
        "CNSHA": "China",
        "CNSHK": "China",
        "CNTAO": "China",
        "CNYTN": "China",
        "COCTG": "Colombia",
        "DEHAM": "Denmark",
        "ESBCN": "Spain",
        "ESVLC": "Spain",
        "GBLGP": "U.K.",
        "GRPIR": "Greece",
        "HKHKG": "Hong Kong",
        "JPUKB": "Japan",
        "KRPUS": "South Korea",
        "LKCMB": "Agg Asia Pacific",
        "MAPTM": "Agg Middle East & Africa",
        "MXZLO": "Mexico",
        "MYPKG": "Agg Asia Pacific",
        "MYTPP": "Agg Asia Pacific",
        "NLRTM": "Netherlands",
        "NZAKL": "Agg Asia Pacific",
        "PAMIT": "Agg Latin America",
        "SAJED": "Agg Middle East & Africa",
        "SAJUB": "Agg Middle East & Africa",
        "SGSIN": "Singapore",
        "THLCH": "Thailand",
        "TWKHH": "Taiwan",
        "USBAL": "U.S.",
        "USCHS": "U.S.",
        "USHOU": "U.S.",
        "USILM": "U.S.",
        "USLAX": "U.S.",
        "USLGB": "U.S.",
        "USMOB": "U.S.",
        "USMSY": "U.S.",
        "USNYC": "U.S.",
        "USORF": "U.S.",
        "USSAV": "U.S.",
        "USTIW": "U.S.",
    }
    # create region column
    rel_df_nona.loc[:, "region"] = rel_df_nona["POL"].apply(lambda x: rel_port_map[x])
    # process retail sales data
    new_cols = [col.strip() for col in sales_data.columns]
    sales_data.columns = new_cols
    # add month column
    sales_data.loc[:, "month"] = sales_data["MonthYear"].apply(
        lambda x: int(x.split("/")[0])
    )
    # add year column
    sales_data.loc[:, "year"] = sales_data["MonthYear"].apply(
        lambda x: int(x.split("/")[1])
    )
    # add date column
    sales_data.loc[:, "date"] = sales_data["MonthYear"].apply(
        lambda x: datetime.datetime.strptime(x, "%m/%Y")
    )
    # TODO: add support for moving average
    sales_data.loc[:, "date(offset)"] = sales_data["date"]


def align_sales(sales_data, rel_df_nona, max_date):
    """Add retail sales column to schedule data

    Args:
        sales_data (pd.DataFrame): retail sales data frame
        rel_df_nona (pd.DataFrame): reliability schedule data frame
        max_date (datetime.datetime): max data threshold
    """

    # create a retail sales map given date and country/region
    # date, country/region -> retail sales index
    regions = [
        "Agg North America",
        "U.S.",
        "Canada",
        "Mexico",
        "Agg Western Europe",
        "Austria",
        "Belgium",
        "Cyprus",
        "Denmark",
        "Euro Area",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Iceland",
        "Ireland",
        "Italy",
        "Luxembourg",
        "Netherlands",
        "Norway",
        "Portugal",
        "Spain",
        "Sweden",
        "Switzerland",
        "U.K.",
        "Agg Asia Pacific",
        "Australia",
        "China",
        "Hong Kong",
        "Indonesia",
        "Japan",
        "Kazakhstan",
        "Macau",
        "Singapore",
        "South Korea",
        "Taiwan",
        "Thailand",
        "Vietnam",
        "Agg Eastern Europe",
        "Bulgaria",
        "Croatia",
        "Czech Republic",
        "Estonia",
        "Hungary",
        "Latvia",
        "Lithuania",
        "Poland",
        "Romania",
        "Russia",
        "Serbia",
        "Slovenia",
        "Turkey",
        "Agg Latin America",
        "Argentina",
        "Brazil",
        "Chile",
        "Colombia",
        "Agg Middle East & Africa",
        "Israel",
        "South Africa",
    ]
    date_region_sales = {}
    for region in regions:
        region_dict = dict(zip(sales_data["date(offset)"], sales_data[region]))
        date_region_sales[region] = region_dict

    # finally, create new columns
    # iterate over rows
    rel_df_nona.loc[:, "retail_sales"] = rel_df_nona.apply(
        lambda x: date_region_sales[x["region"]][x["Date"]]
        if x["Date"] <= max_date
        else None,
        axis=1,
    )


def process_cpi(cpi_df):
    """Add date column to cpi data frame

    Args:
        cpi_df (pd.DataFrame): consumer price index data frame
    """

    cpi_df.columns = [col.strip() for col in cpi_df.columns]

    cpi_df.columns = [
        "MonthYear",
        "Agg North America",
        "U.S.",
        "Canada",
        "Mexico",
        "Agg Western Europe",
        "Austria",
        "Belgium",
        "Cyprus",
        "Denmark",
        "Euro Area",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Iceland",
        "Ireland",
        "Italy",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Norway",
        "Portugal",
        "Spain",
        "Sweden",
        "Switzerland",
        "U.K.",
        "Agg Asia Pacific",
        "Australia",
        "China",
        "India*",
        "Indonesia",
        "Japan",
        "Philippines",
        "Singapore",
        "South Korea",
        "Taiwan",
        "Thailand",
        "Agg Latin America",
        "Argentina",
        "Brazil",
        "Chile",
        "Colombia",
        "Peru",
        "Agg Eastern Europe",
        "Bulgaria",
        "Croatia",
        "Czech Republic",
        "Estonia",
        "Hungary",
        "Latvia",
        "Lithuania",
        "Poland",
        "Romania",
        "Russia",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Turkey",
        "Agg Middle East & Africa",
        "Egypt",
        "Iraq",
        "Israel",
        "South Africa",
    ]

    cpi_df.loc[:, "date"] = cpi_df["MonthYear"].apply(
        lambda x: datetime.datetime.strptime(x, "%m/%Y")
    )

    cpi_df.loc[:, "date(offset)"] = cpi_df["date"]


def align_cpi(cpi_df, rel_df_nona):
    """Align cpi data with schedule data

    Args:
        cpi_df (pd.DataFrame): consumer price index data frame
        rel_df_nona (pd.DataFrame): reliability schdule data frame
    """

    regions_cpi = [
        "Agg North America",
        "U.S.",
        "Canada",
        "Mexico",
        "Agg Western Europe",
        "Austria",
        "Belgium",
        "Cyprus",
        "Denmark",
        "Euro Area",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Iceland",
        "Ireland",
        "Italy",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Norway",
        "Portugal",
        "Spain",
        "Sweden",
        "Switzerland",
        "U.K.",
        "Agg Asia Pacific",
        "Australia",
        "China",
        "India*",
        "Indonesia",
        "Japan",
        "Philippines",
        "Singapore",
        "South Korea",
        "Taiwan",
        "Thailand",
        "Agg Latin America",
        "Argentina",
        "Brazil",
        "Chile",
        "Colombia",
        "Peru",
        "Agg Eastern Europe",
        "Bulgaria",
        "Croatia",
        "Czech Republic",
        "Estonia",
        "Hungary",
        "Latvia",
        "Lithuania",
        "Poland",
        "Romania",
        "Russia",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "Turkey",
        "Agg Middle East & Africa",
        "Egypt",
        "Iraq",
        "Israel",
        "South Africa",
    ]

    date_region_cpi = {}
    for region in regions_cpi:
        region_dict = dict(zip(cpi_df["date(offset)"], cpi_df[region]))

        date_region_cpi[region] = region_dict

    # calculate max date to avoid index error
    max_date = cpi_df["date(offset)"].max()

    rel_df_nona.loc[:, "cpi"] = rel_df_nona.apply(
        lambda x: date_region_cpi[x["region"]][x["Date"]]
        if x["Date"] <= max_date
        else None,
        axis=1,
    )


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

    path = os.path.join(data_dir, filename)
    data = pd.read_excel(path, sheet_name=sheetname)

    return data


def weighted_average_ser(ser: pd.Series):
    """Return weighted average of series

    Args:
        ser (pd.Series): pandas Series with float values

    Returns:
        float: weighted average of series
    """

    wts = pd.Series([1 / val if val != 0 else 0 for val in ser])

    if wts.sum() == 0:
        return 0  # avoid division by zero

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
    return (df["Carrier"] == carrier) & (df["Service"] == service)


def get_reg_train_test(
    df: pd.DataFrame,
    datetime_split: datetime.datetime,
    label="Avg_TTDays",
    use_retail=False,
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
        "Carrier",
        "Service",
        "POD",
        "POL",
        f"{label}_train",
        f"{label}(std)_train",
    ]

    train_min = get_train_wt_avg(train, datetime_split, label=label, agg_fun=np.min)
    train_min.columns = [
        "Carrier",
        "Service",
        "POD",
        "POL",
        f"{label}_min_train",
        f"{label}(std)_min_train",
    ]

    train_max = get_train_wt_avg(train, datetime_split, label=label, agg_fun=np.max)
    train_max.columns = [
        "Carrier",
        "Service",
        "POD",
        "POL",
        f"{label}_max_train",
        f"{label}(std)_max_train",
    ]

    train = train_wt_mean.merge(train, on=["Carrier", "Service", "POD", "POL"])

    train = train_min.merge(train, on=["Carrier", "Service", "POD", "POL"])

    train = train_max.merge(train, on=["Carrier", "Service", "POD", "POL"])

    # val
    val = df[df[date_column] >= datetime_split]

    val = train_wt_mean.merge(val, on=["Carrier", "Service", "POD", "POL"])

    val = train_min.merge(val, on=["Carrier", "Service", "POD", "POL"])

    val = train_max.merge(val, on=["Carrier", "Service", "POD", "POL"])

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
            f"{label}_max_train",
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
            "cpi",
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
    agg_fun=weighted_average_ser,
):
    """Return data frame weighted aggregated indexed by Carrier, Service, POD, and POL columns

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
    train_on_time_rel_by_carr_ser = (
        train[
            [
                "Carrier",
                "Service",
                "POD",
                "POL",
                label,
            ]
        ]
        .groupby(["Carrier", "Service", "POD", "POL"])
        .apply(lambda x: (agg_fun(x[label].values), x[label].values.std()))
        .reset_index()
    )

    train_on_time_rel_by_carr_ser.loc[:, f"{label}"] = train_on_time_rel_by_carr_ser[
        0
    ].apply(lambda x: x[0])
    train_on_time_rel_by_carr_ser.loc[
        :, f"{label}(std)"
    ] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[1])

    train_on_time_rel_by_carr_ser.drop(0, axis=1, inplace=True)

    train_df = train_on_time_rel_by_carr_ser.copy()

    return train_df


def gen_lag(
    df: pd.DataFrame,
    lag=1,
    lag_col="Month(int)",
    target_cols=["OnTime_Reliability"],
    common_cols=["Carrier", "Service", "POL", "POD", "Trade", "Month(int)"],
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
            "Avg_Anchorage_Hours(by_call)_x": "Avg_Anchorage_Hours(by_call)",
        },
        inplace=True,
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
    label="Avg_TTDays",
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
        train_preds = list(
            map(lambda x: 100 if x >= 100 else x, model.predict(train_X_rg))
        )
        val_preds = list(map(lambda x: 100 if x >= 100 else x, model.predict(val_X_rg)))

        train_preds = list(map(lambda x: 0 if x <= 0 else x, train_preds))
        val_preds = list(map(lambda x: 0 if x <= 0 else x, val_preds))

    # val metrics
    val_X, val_gt, preds_array = filter_nonzero_values(val_X, val_y, val_preds, label)
    val_mae = mean_absolute_error(val_gt, preds_array)
    val_mape = mean_absolute_percentage_error(val_gt, preds_array)

    # train metrics
    train_X, train_gt, train_preds_array = filter_nonzero_values(
        train_X, train_y, train_preds, label
    )
    train_mae = mean_absolute_error(train_gt, train_preds_array)
    train_mape = mean_absolute_percentage_error(train_gt, train_preds_array)

    # create error result dataframe
    result_df = val_X.copy()
    result_df.loc[:, "actual"] = val_y
    result_df.loc[:, "pred"] = preds_array
    result_df.loc[:, "error"] = preds_array - val_y
    result_df.loc[:, "perc_error"] = (preds_array - val_y) / val_y

    result_df = result_df[
        ["Carrier", "Service", "POD", "POL", "actual", "pred", "error", "perc_error"]
    ]

    # overestimate mask
    diff = preds_array - val_y
    mask = diff > 0

    # mape
    if include_overestimates:

        if sum(mask) == 0:
            raise Exception("There are not overestimated preds!")

        # compute mae (over)
        val_mae_over = diff[mask].mean()
        # series mask
        mask_ser = mask.reset_index()[label]
        # compute mape (over)
        val_preds_over = pd.Series(preds_array)[mask_ser]
        val_y_over = pd.Series(list(val_y))[mask_ser]
        val_mape_over = mean_absolute_percentage_error(val_y_over, val_preds_over)

        return (
            train_mae,
            train_mape,
            val_mae,
            val_mape,
            val_mae_over,
            val_mape_over,
            result_df,
        )

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
    common_cols = list(set(train_X_rg.columns).intersection(set(val_X_rg.columns)))

    return train_X_rg[common_cols], val_X_rg[common_cols]
