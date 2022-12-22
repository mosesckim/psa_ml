import datetime

import pandas as pd


# process data
def process_schedule_data(schedule_data):
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


# restrict to carrier service routes with all months covered
def restrict_by_coverage(rel_df_nona, min_no_months=9):

    rel_df_nona_cvg = rel_df_nona.groupby(
        ["POL", "POD", "Carrier", "Service"]
    ).apply(lambda x: len(x["Month"].unique())
    )

    rel_df_nona_full_cvg = rel_df_nona_cvg[rel_df_nona_cvg==min_no_months]  # TODO: REMOVE hardcoded value

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


def split_data(rel_df_nona, datetime_split, label="Avg_TTDays"):

    # train
    train = rel_df_nona[rel_df_nona["Date"] < datetime_split]
    # val
    val = rel_df_nona[rel_df_nona["Date"] >= datetime_split]

    # let's get multi-index pairs from train
    train_indices = list(
        train[
            ["Carrier", "Service", "POD", "POL", label]
        ].groupby(["Carrier", "Service", "POD", "POL"]).mean().index
    )

    # now find the intersection between train an val
    indices_inter = []
    for ind, row in val.iterrows():
        ind_pair = (row["Carrier"], row["Service"], row["POD"], row["POL"])
        if ind_pair in train_indices:
            indices_inter.append(ind)

    # now restrict to the indices in the intersection
    val_res = val.loc[indices_inter, :]

    # use weighted average
    train_on_time_rel_by_carr_ser = train[[
        "Carrier", "Service", "POD", "POL", label
    ]].groupby(["Carrier", "Service", "POD", "POL"]).apply(
        lambda x: (weighted_average_ser(x[label].values), x[label].values.std())
    ).reset_index()

    train_on_time_rel_by_carr_ser.loc[:, f"{label}"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[0])
    train_on_time_rel_by_carr_ser.loc[:, f"{label}(std)"] = train_on_time_rel_by_carr_ser[0].apply(lambda x: x[1])

    train_on_time_rel_by_carr_ser.drop(0, axis=1, inplace=True)

    train_df = train_on_time_rel_by_carr_ser.copy()

    return train_df, val_res


def weighted_average_ser(ser):

    wts = pd.Series([1 / val if val != 0 else 0 for val in ser])

    if wts.sum() == 0: return 0

    return (ser * wts).sum() / wts.sum()


def get_carr_serv_mask(df, carrier, service):

    return (df["Carrier"]==carrier) & \
        (df["Service"]==service)