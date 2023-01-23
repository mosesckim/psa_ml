# generic evaluation script
# given model
# trade lane
# and/or set of routes

# also add delta on macroecon features

import datetime
import argparse
import yaml
import os

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from ontime.src.models import BaselineModel
from ontime.src.utils import split_data, process_schedule_data, restrict_by_coverage, \
    get_carr_serv_mask, get_reg_train_test, compute_eval_metrics, load_excel_data


def main():

    parser = argparse.ArgumentParser()
    # add args
    parser.add_argument(
        "-d", "--data_dir_path", default="ontime/data"
    )
    parser.add_argument(
        "--split_month", default=8
    )
    parser.add_argument(
        "--max_month", default=9
    )
    parser.add_argument(
        "--label", default="Avg_TTDays"
    )
    parser.add_argument(
        "--partial_pred", default=False
    )
    parser.add_argument(
        "--overall_pred", default=True
    )
    parser.add_argument(
        "--restrict_trade", default=False
    )
    parser.add_argument(
        "--trade_option", default="Asia-North America West Coast"
    )
    parser.add_argument(
        "--carrier_option", default="ANL"
    )
    parser.add_argument(
        "--service_option", default="EPIC"
    )
    parser.add_argument(
        "--eval_lin_reg", default=False
    )
    parser.add_argument(
        "--include_reg", default=True
    )
    parser.add_argument(
        "--schedule_filename",
        default="2022-11-29TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv"
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

    # DATA
    config = {
        "data_path": "ontime/data",
        "port_call": {
            "filename": "IHS_PORT_PERFORMANCE.xlsx",
            "sheet": "Sheet1"
        },
        "sales": {
            "filename": "Retail Sales 202210.xlsx",
            "sheet": "Sales"
        },
        "cpi": {
            "filename": "CPI Core 202210.xlsx",
            "sheet": "Core CPI"
        },
        "air_freight": {
            "filename": "AirFrieght total Rate USD per 1000kg Shanghai to Los angeles.xlsx",
            "sheet": "Sheet1"
        }
    }

    print("Loading data...")

    # shipping schedule data
    rel_path = os.path.join(
        data_dir,
        schedule_filename
    )
    schedule_data = pd.read_csv(rel_path)
    # port call data
    port_data = load_excel_data(config, "port_call")
    # air freight data (shanghai - lax)
    air_freight_data = load_excel_data(config, "air_freight")
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
        rel_df_nona = rel_df_nona[rel_df_nona["Trade"]==trade_option]

    # port call
    # seaport code dict (schedule -> port call)
    seaport_code_map= {"CNSHG": "CNSHA", "CNTNJ": "CNTXG", "CNQIN": "CNTAO"}
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
    agg_cols = ["seaport_code", "Month", "Year"]
    target_cols = ["Total_Calls", "Port_Hours", "Anchorage_Hours"]
    # sum up calls, port/anchorage hours
    # and aggregate by port, month, and year
    port_hours_avg = port_call_df[target_cols + agg_cols].groupby(
        agg_cols
    ).sum().reset_index()
    # average port hours by port, month
    port_hours_avg.loc[:, "Avg_Port_Hours(by_call)"] = port_hours_avg[
        "Port_Hours"
    ] / port_hours_avg["Total_Calls"]
    # average anchorage hours by port, month
    port_hours_avg.loc[:, "Avg_Anchorage_Hours(by_call)"] = port_hours_avg[
        "Anchorage_Hours"
    ] / port_hours_avg["Total_Calls"]
    # merge avg hours
    rel_df_no_orf_pt_hrs = rel_df_no_orf.merge(
        port_hours_avg,
        left_on=["Calendary_Year", "Month(int)", "seaport_code"],
        right_on=["Year", "Month", "seaport_code"]
    )

    # retail sales
    # reliability POL mapping -> retail_sales country/region
    rel_port_map = {
        'AEAUH': 'Agg Middle East & Africa',
        'AEJEA': 'Agg Middle East & Africa',
        'BEANR': 'Belgium',
        'BRRIG': 'Brazil',
        'CNNGB': 'China',
        'CNSHA': 'China',
        'CNSHK': 'China',
        'CNTAO': 'China',
        'CNYTN': 'China',
        'COCTG': 'Colombia',
        'DEHAM': 'Denmark',
        'ESBCN': 'Spain',
        'ESVLC': 'Spain',
        'GBLGP': 'U.K.',
        'GRPIR': 'Greece',
        'HKHKG': 'Hong Kong',
        'JPUKB': 'Japan',
        'KRPUS': 'South Korea',
        'LKCMB': 'Agg Asia Pacific',
        'MAPTM': 'Agg Middle East & Africa',
        'MXZLO': 'Mexico',
        'MYPKG': 'Agg Asia Pacific',
        'MYTPP': 'Agg Asia Pacific',
        'NLRTM': 'Netherlands',
        'NZAKL': 'Agg Asia Pacific',
        'PAMIT': 'Agg Latin America',
        'SAJED': 'Agg Middle East & Africa',
        'SAJUB': 'Agg Middle East & Africa',
        'SGSIN': 'Singapore',
        'THLCH': 'Thailand',
        'TWKHH': 'Taiwan',
        'USBAL': 'U.S.',
        'USCHS': 'U.S.',
        'USHOU': 'U.S.',
        'USILM': 'U.S.',
        'USLAX': 'U.S.',
        'USLGB': 'U.S.',
        'USMOB': 'U.S.',
        'USMSY': 'U.S.',
        'USNYC': 'U.S.',
        'USORF': 'U.S.',
        'USSAV': 'U.S.',
        'USTIW': 'U.S.'
    }
    # create region column
    rel_df_nona.loc[:, "region"] = rel_df_nona["POL"].apply(
        lambda x: rel_port_map[x]
    )
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
        lambda x: datetime.datetime.strptime(
            x, "%m/%Y"
        )
    )
    # TODO: add support for moving average
    sales_data.loc[:, "date(offset)"] = sales_data['date']
    # create a retail sales map given date and country/region
    # date, country/region -> retail sales index
    regions = [
        'Agg North America', 'U.S.', 'Canada', 'Mexico',
        'Agg Western Europe', 'Austria', 'Belgium', 'Cyprus', 'Denmark',
        'Euro Area', 'Finland', 'France', 'Germany', 'Greece', 'Iceland',
        'Ireland', 'Italy', 'Luxembourg', 'Netherlands', 'Norway', 'Portugal',
        'Spain', 'Sweden', 'Switzerland', 'U.K.', 'Agg Asia Pacific',
        'Australia', 'China', 'Hong Kong', 'Indonesia', 'Japan', 'Kazakhstan',
        'Macau', 'Singapore', 'South Korea', 'Taiwan', 'Thailand', 'Vietnam',
        'Agg Eastern Europe', 'Bulgaria', 'Croatia', 'Czech Republic',
        'Estonia', 'Hungary', 'Latvia', 'Lithuania', 'Poland', 'Romania',
        'Russia', 'Serbia', 'Slovenia', 'Turkey', 'Agg Latin America',
        'Argentina', 'Brazil', 'Chile', 'Colombia', 'Agg Middle East & Africa',
        'Israel', 'South Africa'
    ]
    date_region_sales = {}
    for region in regions:
        region_dict = dict(
            zip(
                sales_data["date(offset)"],
                sales_data[region]
            )
        )
        date_region_sales[region] = region_dict

    # TODO: calculate max date given all datasets to be merged
    # calculate max date to avoid index error
    max_date = sales_data["date(offset)"].max()
    # finally, create new columns
    # iterate over rows
    rel_df_nona.loc[:, "retail_sales"] = rel_df_nona.apply(
        lambda x: date_region_sales[x["region"]][x["Date"]] if x["Date"] <= max_date else None, axis=1
    )


    # PREPARE DATA
    datetime_split = datetime.datetime(2022, split_month, 1)

    # train_df rows have unique POD, POL, Carrier, Service columns values
    # val_res rows don't  TODO: add this in func description
    train_df, val_res = split_data(rel_df_nona, datetime_split, max_month=max_date.month)

    val_X = val_res[["Carrier", "Service", "POD", "POL"]]
    val_y = val_res[label]

    train_df_filtered = train_df.copy()
    val_X_filtered = val_X.copy()
    val_y_filtered = val_y.copy()


    # EVALUATION

    if partial_pred:
        # train
        train_mask = get_carr_serv_mask(train_df_filtered, carrier_option, service_option)
        train_df_filtered = train_df_filtered[train_mask]
        # val
        val_mask = get_carr_serv_mask(val_X_filtered, carrier_option, service_option)
        val_X_filtered = val_X_filtered[val_mask]
        val_y_filtered = val_y_filtered[val_mask]

    if val_X_filtered.shape[0] == 0 or train_df_filtered.shape[0] == 0:
        raise Exception('Insufficient data, pease choose another split')


    # linear regression
    if include_reg and overall_pred:

        # TODO: automate date upper threshold given my macroeconimc feature data set
        val_res = val_res[val_res["Date"] < datetime.datetime(2022, 9, 1)]

        split_month = min(8, split_month)
        datetime_split = datetime.datetime(2022, split_month, 1)

        # linear regression split (retail)
        train_X_rg_ret, train_y_rg_ret, val_X_rg_ret, val_y_rg_ret = get_reg_train_test(
            rel_df_nona,
            datetime_split,
            label=label,
            use_retail=True
        )

        # # TODO: include in pytest
        # print("val sales shape: ", val_X_rg_ret.shape)
        # print("val res shape: ", val_res.shape)

        try:
            # evaluate linear regression
            linreg = LinearRegression()

            train_mae_rg_ret, train_mape_rg_ret, \
                val_mae_rg_ret, val_mape_rg_ret, val_mae_over_rg_ret, val_mape_over_rg_ret, \
                    result_df_rg_ret = compute_eval_metrics(
                linreg,
                train_X_rg_ret,
                val_X_rg_ret,
                train_y_rg_ret,
                val_y_rg_ret,
                include_overestimates=True,
                label=label
            )

            eval_lin_reg = True

        except:
            raise Exception("Not enough data. Choose a different split")


        # instantiate baseline model
        base_model = BaselineModel()
        base_model.fit(train_df_filtered, label)

        preds, preds_std = base_model.predict(val_X_filtered)
        preds_array, preds_std_array = list(map(lambda x: x.values, [preds, preds_std]))


        # # prediction by row
        # preds = []
        # preds_std = []
        # print("Computing predictions...")  # TODO: progress bar

        # val_X_filtered = val_res[["Carrier", "Service", "POD", "POL"]]
        # val_y_filtered = val_res[label]

        # for ind, row in val_X_filtered.iterrows():
        #     pred, pred_std = base_model.predict_(*row)

        #     preds.append(pred)
        #     preds_std.append(pred_std)
        # preds_array = np.array(preds)
        # preds_std_array = np.array(preds_std)


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
                mape_under = mean_absolute_percentage_error(val_y_values_under, preds_array_under)
                mae_under = round(mae_under, 3)
                mape_under = round(mape_under, 3)
            else:
                mae_under = "NA"
                mape_under = "NA"


            print("Predictions")
            df_preds = val_X_filtered.copy()
            df_preds.loc[:, "actual"] = val_y_filtered
            df_preds.loc[:, "pred"] = preds_array
            df_preds.loc[:, "error"] = preds_array - val_y_filtered
            df_preds.loc[:, "perc_error"] = (preds - val_y_filtered) / val_y_filtered

    if eval_lin_reg:
        print("macro mape: ", val_mape_rg_ret)
    else:
        raise Exception("Not enough data. Choose a different split")

    print("baseline mape: ", baseline_mape)



if __name__ == "__main__":


    # CONFIG
    with open("ontime/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # data dir
    data_dir = config["data_path"]


    # other params
    eval_params = config["eval"]

    split_month = eval_params["split_month"]
    max_month = eval_params["max_month"]
    label = eval_params["label"]
    partial_pred = eval_params["partial_pred"]
    overall_pred = eval_params["overall_pred"]

    restrict_trade = eval_params["restrict_trade"]
    trade_option = eval_params["trade_option"]
    carrier_option = eval_params["carrier_option"]
    service_option = eval_params["service_option"]

    eval_lin_reg = eval_params["eval_lin_reg"]
    include_reg = eval_params["include_reg"]

    # DATA

    print("Loading data...")

    # shipping schedule data
    rel_path = os.path.join(
        data_dir,
        config["schedule"]["filename"]
    )
    schedule_data = pd.read_csv(rel_path)
    # port call data
    port_data = load_excel_data(config, "port_call")
    # air freight data (shanghai - lax)
    air_freight_data = load_excel_data(config, "air_freight")
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
    import pdb; pdb.set_trace()
    # restrict trade
    if restrict_trade:
        rel_df_nona = rel_df_nona[rel_df_nona["Trade"]==trade_option]

    # port call
    # seaport code dict (schedule -> port call)
    seaport_code_map= {"CNSHG": "CNSHA", "CNTNJ": "CNTXG", "CNQIN": "CNTAO"}
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
    agg_cols = ["seaport_code", "Month", "Year"]
    target_cols = ["Total_Calls", "Port_Hours", "Anchorage_Hours"]
    # sum up calls, port/anchorage hours
    # and aggregate by port, month, and year
    port_hours_avg = port_call_df[target_cols + agg_cols].groupby(
        agg_cols
    ).sum().reset_index()
    # average port hours by port, month
    port_hours_avg.loc[:, "Avg_Port_Hours(by_call)"] = port_hours_avg[
        "Port_Hours"
    ] / port_hours_avg["Total_Calls"]
    # average anchorage hours by port, month
    port_hours_avg.loc[:, "Avg_Anchorage_Hours(by_call)"] = port_hours_avg[
        "Anchorage_Hours"
    ] / port_hours_avg["Total_Calls"]
    # merge avg hours
    rel_df_no_orf_pt_hrs = rel_df_no_orf.merge(
        port_hours_avg,
        left_on=["Calendary_Year", "Month(int)", "seaport_code"],
        right_on=["Year", "Month", "seaport_code"]
    )

    # retail sales
    # reliability POL mapping -> retail_sales country/region
    rel_port_map = {
        'AEAUH': 'Agg Middle East & Africa',
        'AEJEA': 'Agg Middle East & Africa',
        'BEANR': 'Belgium',
        'BRRIG': 'Brazil',
        'CNNGB': 'China',
        'CNSHA': 'China',
        'CNSHK': 'China',
        'CNTAO': 'China',
        'CNYTN': 'China',
        'COCTG': 'Colombia',
        'DEHAM': 'Denmark',
        'ESBCN': 'Spain',
        'ESVLC': 'Spain',
        'GBLGP': 'U.K.',
        'GRPIR': 'Greece',
        'HKHKG': 'Hong Kong',
        'JPUKB': 'Japan',
        'KRPUS': 'South Korea',
        'LKCMB': 'Agg Asia Pacific',
        'MAPTM': 'Agg Middle East & Africa',
        'MXZLO': 'Mexico',
        'MYPKG': 'Agg Asia Pacific',
        'MYTPP': 'Agg Asia Pacific',
        'NLRTM': 'Netherlands',
        'NZAKL': 'Agg Asia Pacific',
        'PAMIT': 'Agg Latin America',
        'SAJED': 'Agg Middle East & Africa',
        'SAJUB': 'Agg Middle East & Africa',
        'SGSIN': 'Singapore',
        'THLCH': 'Thailand',
        'TWKHH': 'Taiwan',
        'USBAL': 'U.S.',
        'USCHS': 'U.S.',
        'USHOU': 'U.S.',
        'USILM': 'U.S.',
        'USLAX': 'U.S.',
        'USLGB': 'U.S.',
        'USMOB': 'U.S.',
        'USMSY': 'U.S.',
        'USNYC': 'U.S.',
        'USORF': 'U.S.',
        'USSAV': 'U.S.',
        'USTIW': 'U.S.'
    }
    # create region column
    rel_df_nona.loc[:, "region"] = rel_df_nona["POL"].apply(
        lambda x: rel_port_map[x]
    )
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
        lambda x: datetime.datetime.strptime(
            x, "%m/%Y"
        )
    )
    # TODO: add support for moving average
    sales_data.loc[:, "date(offset)"] = sales_data['date']
    # create a retail sales map given date and country/region
    # date, country/region -> retail sales index
    regions = [
        'Agg North America', 'U.S.', 'Canada', 'Mexico',
        'Agg Western Europe', 'Austria', 'Belgium', 'Cyprus', 'Denmark',
        'Euro Area', 'Finland', 'France', 'Germany', 'Greece', 'Iceland',
        'Ireland', 'Italy', 'Luxembourg', 'Netherlands', 'Norway', 'Portugal',
        'Spain', 'Sweden', 'Switzerland', 'U.K.', 'Agg Asia Pacific',
        'Australia', 'China', 'Hong Kong', 'Indonesia', 'Japan', 'Kazakhstan',
        'Macau', 'Singapore', 'South Korea', 'Taiwan', 'Thailand', 'Vietnam',
        'Agg Eastern Europe', 'Bulgaria', 'Croatia', 'Czech Republic',
        'Estonia', 'Hungary', 'Latvia', 'Lithuania', 'Poland', 'Romania',
        'Russia', 'Serbia', 'Slovenia', 'Turkey', 'Agg Latin America',
        'Argentina', 'Brazil', 'Chile', 'Colombia', 'Agg Middle East & Africa',
        'Israel', 'South Africa'
    ]
    date_region_sales = {}
    for region in regions:
        region_dict = dict(
            zip(
                sales_data["date(offset)"],
                sales_data[region]
            )
        )
        date_region_sales[region] = region_dict

    # TODO: calculate max date given all datasets to be merged
    # calculate max date to avoid index error
    max_date = sales_data["date(offset)"].max()
    # finally, create new columns
    # iterate over rows
    rel_df_nona.loc[:, "retail_sales"] = rel_df_nona.apply(
        lambda x: date_region_sales[x["region"]][x["Date"]] if x["Date"] <= max_date else None, axis=1
    )


    # PREPARE DATA
    datetime_split = datetime.datetime(2022, split_month, 1)

    # train_df rows have unique POD, POL, Carrier, Service columns values
    # val_res rows don't  TODO: add this in func description
    train_df, val_res = split_data(rel_df_nona, datetime_split, max_month=max_date.month)
    import pdb; pdb.set_trace()

    val_X = val_res[["Carrier", "Service", "POD", "POL"]]
    val_y = val_res[label]

    train_df_filtered = train_df.copy()
    val_X_filtered = val_X.copy()
    val_y_filtered = val_y.copy()


    # EVALUATION

    if partial_pred:
        # train
        train_mask = get_carr_serv_mask(train_df_filtered, carrier_option, service_option)
        train_df_filtered = train_df_filtered[train_mask]
        # val
        val_mask = get_carr_serv_mask(val_X_filtered, carrier_option, service_option)
        val_X_filtered = val_X_filtered[val_mask]
        val_y_filtered = val_y_filtered[val_mask]

    if val_X_filtered.shape[0] == 0 or train_df_filtered.shape[0] == 0:
        raise Exception('Insufficient data, pease choose another split')


    # linear regression
    if include_reg and overall_pred:

        # TODO: automate date upper threshold given my macroeconimc feature data set
        val_res = val_res[val_res["Date"] < datetime.datetime(2022, 9, 1)]

        split_month = min(8, split_month)
        datetime_split = datetime.datetime(2022, split_month, 1)

        # linear regression split (retail)
        train_X_rg_ret, train_y_rg_ret, val_X_rg_ret, val_y_rg_ret = get_reg_train_test(
            rel_df_nona,
            datetime_split,
            label=label,
            use_retail=True
        )

        # # TODO: include in pytest
        # print("val sales shape: ", val_X_rg_ret.shape)
        # print("val res shape: ", val_res.shape)

        try:
            # evaluate linear regression
            linreg = LinearRegression()

            train_mae_rg_ret, train_mape_rg_ret, \
                val_mae_rg_ret, val_mape_rg_ret, val_mae_over_rg_ret, val_mape_over_rg_ret, \
                    result_df_rg_ret = compute_eval_metrics(
                linreg,
                train_X_rg_ret,
                val_X_rg_ret,
                train_y_rg_ret,
                val_y_rg_ret,
                include_overestimates=True,
                label=label
            )

            eval_lin_reg = True

        except:
            raise Exception("Not enough data. Choose a different split")


        # instantiate baseline model
        base_model = BaselineModel()
        base_model.fit(train_df_filtered, label)

        preds, preds_std = base_model.predict(val_X_filtered)
        preds_array, preds_std_array = list(map(lambda x: x.values, [preds, preds_std]))


        # # prediction by row
        # preds = []
        # preds_std = []
        # print("Computing predictions...")  # TODO: progress bar

        # val_X_filtered = val_res[["Carrier", "Service", "POD", "POL"]]
        # val_y_filtered = val_res[label]

        # for ind, row in val_X_filtered.iterrows():
        #     pred, pred_std = base_model.predict_(*row)

        #     preds.append(pred)
        #     preds_std.append(pred_std)
        # preds_array = np.array(preds)
        # preds_std_array = np.array(preds_std)


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
                mape_under = mean_absolute_percentage_error(val_y_values_under, preds_array_under)
                mae_under = round(mae_under, 3)
                mape_under = round(mape_under, 3)
            else:
                mae_under = "NA"
                mape_under = "NA"


            print("Predictions")
            df_preds = val_X_filtered.copy()
            df_preds.loc[:, "actual"] = val_y_filtered
            df_preds.loc[:, "pred"] = preds_array
            df_preds.loc[:, "error"] = preds_array - val_y_filtered
            df_preds.loc[:, "perc_error"] = (preds - val_y_filtered) / val_y_filtered

    if eval_lin_reg:
        print("macro mape: ", val_mape_rg_ret)
    else:
        raise Exception("Not enough data. Choose a different split")

    print("baseline mape: ", baseline_mape)
