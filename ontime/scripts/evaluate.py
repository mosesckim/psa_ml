import datetime
import argparse
import os

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from ontime.src.models import BaselineModel
from ontime.src.utils import (
    split_data,
    process_schedule_data,
    restrict_by_coverage,
    get_carr_serv_mask,
    get_reg_train_test,
    compute_eval_metrics,
    load_excel_data,
    align_port_call,
    process_sales,
    align_sales,
    process_cpi,
    align_cpi,
)

pd.set_option("mode.chained_assignment", None)


def main():

    parser = argparse.ArgumentParser()
    # add args
    parser.add_argument("-d", "--data_dir_path", default="ontime/data")
    parser.add_argument("--split_month", default=8)
    parser.add_argument("--max_month", default=9)
    parser.add_argument("--label", default="Avg_TTDays")
    parser.add_argument("--partial_pred", default=False)
    parser.add_argument("--overall_pred", default=True)
    parser.add_argument("--restrict_trade", default=True)
    parser.add_argument("--trade_option", default="Asia-Europe") #"Asia-North America West Coast")
    parser.add_argument("--carrier_option", default="ANL")
    parser.add_argument("--service_option", default="EPIC")
    parser.add_argument("--eval_lin_reg", default=False)
    parser.add_argument("--include_reg", default=True)
    parser.add_argument(
        "--schedule_filename",
        default="2022-11-29TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv",
    )

    # parse
    args = parser.parse_args()
    # data dir
    data_dir = args.data_dir_path
    # other params
    split_month = args.split_month
    label = args.label
    partial_pred = args.partial_pred
    overall_pred = args.overall_pred

    restrict_trade = args.restrict_trade
    trade_option = args.trade_option
    carrier_option = args.carrier_option
    service_option = args.service_option

    eval_lin_reg = args.eval_lin_reg
    include_reg = args.include_reg

    schedule_filename = args.schedule_filename

    # TODO: add config params to parser
    # DATA
    config = {
        "data_path": "ontime/data",
        "port_call": {"filename": "IHS_PORT_PERFORMANCE.xlsx", "sheet": "Sheet1"},
        "sales": {"filename": "Retail Sales 202210.xlsx", "sheet": "Sales"},
        "cpi": {"filename": "CPI Core 202210.xlsx", "sheet": "Core CPI"},
        "air_freight": {
            "filename": "AirFrieght total Rate USD per 1000kg Shanghai to Los angeles.xlsx",
            "sheet": "Sheet1",
        },
    }

    print("Loading data...")

    # shipping schedule data
    rel_path = os.path.join(data_dir, schedule_filename)
    schedule_data = pd.read_csv(rel_path)
    # # port call data
    # port_data = load_excel_data(config, "port_call")
    # # air freight data (shanghai - lax)
    # air_freight_data = load_excel_data(config, "air_freight")

    # retail sales
    sales_data = load_excel_data(config, "sales")
    # cpi
    cpi_data = load_excel_data(config, "cpi")

    # PREPROCESS DATA

    # reliability schedule
    # add date column, process strings, etc.
    rel_df_nona = process_schedule_data(schedule_data)
    # restrict by coverage
    rel_df_nona = restrict_by_coverage(rel_df_nona)

    # restrict trade
    if restrict_trade:
        rel_df_nona = rel_df_nona[rel_df_nona["Trade"] == trade_option]

    # retail sales
    process_sales(sales_data, rel_df_nona)

    # TODO: calculate max date given all datasets to be merged
    # calculate max date to avoid index error
    max_date = sales_data["date(offset)"].max()
    align_sales(sales_data, rel_df_nona, max_date)

    # cpi
    process_cpi(cpi_data)
    align_cpi(cpi_data, rel_df_nona)

    # PREPARE DATA
    datetime_split = datetime.datetime(2022, split_month, 1)

    # train_df rows have unique POD, POL, Carrier, Service columns values
    # val_res rows don't  TODO: add this in func description
    train_df, val_res = split_data(
        rel_df_nona, datetime_split, max_month=max_date.month
    )

    val_X = val_res[["Carrier", "Service", "POD", "POL"]]
    val_y = val_res[label]

    train_df_filtered = train_df.copy()
    val_X_filtered = val_X.copy()
    val_y_filtered = val_y.copy()

    # EVALUATION

    if partial_pred:
        # train
        train_mask = get_carr_serv_mask(
            train_df_filtered, carrier_option, service_option
        )
        train_df_filtered = train_df_filtered[train_mask]
        # val
        val_mask = get_carr_serv_mask(val_X_filtered, carrier_option, service_option)
        val_X_filtered = val_X_filtered[val_mask]
        val_y_filtered = val_y_filtered[val_mask]

    if val_X_filtered.shape[0] == 0 or train_df_filtered.shape[0] == 0:
        raise Exception("Insufficient data, pease choose another split")

    # linear regression
    if include_reg and overall_pred:

        # instantiate baseline model
        base_model = BaselineModel()
        base_model.fit(train_df_filtered, label)

        preds = []
        preds_std = []
        for ind, row in val_X_filtered.iterrows():
            pred, pred_std = base_model.predict_(*row)

            preds.append(pred)
            preds_std.append(pred_std)

        preds_array = np.array(preds)
        preds_std_array = np.array(preds_std)

        # parallelize prediction method
        # preds, preds_std = base_model.predict(val_X_filtered)
        # preds_array, preds_std_array = list(map(lambda x: x.values, [preds, preds_std]))

        nonzero_mask = val_y_filtered != 0  # for mape computation
        nonzero_mask = nonzero_mask.reset_index()[label]

        if sum(nonzero_mask) != 0:

            preds = pd.Series(preds)[nonzero_mask]
            preds_std = pd.Series(preds_std)[nonzero_mask]

            val_y_filtered = val_y_filtered.reset_index()[label]
            val_y_filtered = val_y_filtered[nonzero_mask]

            val_X_filtered = val_X_filtered.reset_index().drop("index", axis=1)
            val_X_filtered = val_X_filtered[nonzero_mask]

            preds_array = np.array(preds)
            preds_std_array = np.array(preds_std)

            val_gt = val_y_filtered.values

            baseline_mae = mean_absolute_error(val_gt, preds_array)
            baseline_mape = mean_absolute_percentage_error(val_gt, preds_array)

            # calculate mape underestimates
            diff = preds_array - val_gt
            mask = diff < 0

            if sum(mask) != 0:
                preds_array_under = preds_array[mask]
                val_y_values_under = val_gt[mask]
                mae_under = mean_absolute_error(preds_array_under, val_y_values_under)
                mape_under = mean_absolute_percentage_error(
                    val_y_values_under, preds_array_under
                )
                mae_under = round(mae_under, 3)
                mape_under = round(mape_under, 3)
            else:
                mae_under = "NA"
                mape_under = "NA"

            print(f"Predictions: {trade_option}")
            df_preds = val_X_filtered.copy()
            df_preds.loc[:, "actual"] = val_y_filtered
            df_preds.loc[:, "pred"] = preds_array
            df_preds.loc[:, "error"] = preds_array - val_y_filtered
            df_preds.loc[:, "perc_error"] = (preds - val_y_filtered) / val_y_filtered




        # TODO: automate date upper threshold given my macroeconimc feature data set
        val_res = val_res[val_res["Date"] < datetime.datetime(2022, 9, 1)]

        split_month = min(8, split_month)
        datetime_split = datetime.datetime(2022, split_month, 1)

        # linear regression split (retail)
        train_X_rg_ret, train_y_rg_ret, val_X_rg_ret, val_y_rg_ret = get_reg_train_test(
            rel_df_nona, datetime_split, label=label, use_retail=True
        )

        # # TODO: include in pytest
        # print("val sales shape: ", val_X_rg_ret.shape)
        # print("val res shape: ", val_res.shape)

        try:
            # evaluate linear regression
            linreg = LinearRegression()

            (
                train_mae_rg_ret,
                train_mape_rg_ret,
                val_mae_rg_ret,
                val_mape_rg_ret,
                val_mae_over_rg_ret,
                val_mape_over_rg_ret,
                result_df_rg_ret,
            ) = compute_eval_metrics(
                linreg,
                train_X_rg_ret,
                val_X_rg_ret,
                train_y_rg_ret,
                val_y_rg_ret,
                include_overestimates=True,
                label=label,
            )

            eval_lin_reg = True

            # random forests
            rf = RandomForestRegressor()

            (
                train_mae_rf_ret,
                train_mape_rf_ret,
                val_mae_rf_ret,
                val_mape_rf_ret,
                val_mae_over_rf_ret,
                val_mape_over_rf_ret,
                result_df_rf_ret,
            ) = compute_eval_metrics(
                rf,
                train_X_rg_ret,
                val_X_rg_ret,
                train_y_rg_ret,
                val_y_rg_ret,
                include_overestimates=True,
                label=label,
            )


        except:
            raise Exception("Not enough data. Choose a different split")

    if eval_lin_reg:
        print("macro mape (linear regression): ", val_mape_rg_ret)

        print("macro mape (random forest): ", val_mape_rf_ret)
    else:
        raise Exception("Not enough data. Choose a different split")

    print("baseline mape: ", baseline_mape)


if __name__ == "__main__":

    main()
