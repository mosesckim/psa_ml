# generic evaluation script
# given model
# trade lane
# and/or set of routes

import datetime
import yaml
import os

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from ontime.src.models import BaselineModel
from ontime.src.utils import split_data, process_schedule_data, restrict_by_coverage, \
    get_carr_serv_mask, get_reg_train_test, compute_train_val_mae


if __name__ == "__main__":


    # CONFIG

    with open("ontime/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # data dir
    data_dir = config["data_path"]


    # other params
    eval_params = config["eval"]

    split_month = eval_params["split_month"]
    label = eval_params["label"]
    partial_pred = eval_params["partial_pred"]
    overall_pred = eval_params["overall_pred"]

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
    port_filename = config["port_call"]["filename"]
    port_sheetname = config["port_call"]["sheet"]
    port_path = os.path.join(
        data_dir,
        port_filename
    )
    port_data = pd.read_excel(
        port_path,
        sheet_name=port_sheetname
    )

    # air freight data (shanghai - lax)
    air_freight_filename = config["air_freight"]["filename"]
    air_freight_sheetname = config["air_freight"]["sheet"]
    air_freight_path = os.path.join(
        data_dir,
        air_freight_filename
    )
    air_freight_data = pd.read_excel(
        air_freight_path,
        sheet_name=air_freight_sheetname
    )

    # retail sales
    sales_filename = config["sales"]["filename"]
    sales_sheetname = config["sales"]["sheet"]
    sales_path = os.path.join(
        data_dir,
        sales_filename
    )
    sales_data = pd.read_excel(
        sales_path,
        sheet_name=sales_sheetname
    )

    # cpi
    cpi_filename = config["cpi"]["filename"]
    cpi_sheetname = config["cpi"]["sheet"]
    cpi_path = os.path.join(
        data_dir,
        cpi_filename
    )
    cpi_data = pd.read_excel(
        cpi_path,
        sheet_name=cpi_sheetname
    )

    # PREPROCESS DATA

    # RELIABILITY SCHEDULE
    rel_df_nona = process_schedule_data(schedule_data)
    # rel_df_nona = restrict_by_coverage(rel_df_nona)  # need to?

    datetime_split = datetime.datetime(2022, split_month, 1)
    train_df, val_res = split_data(rel_df_nona, datetime_split, label=label)


    val_X = val_res[["Carrier", "Service", "POD", "POL"]]
    val_y = val_res[label]

    train_df_filtered = train_df.copy()
    val_X_filtered = val_X.copy()
    val_y_filtered = val_y.copy()


    # port call
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

    port_hours_avg_2022 = port_hours_avg[port_hours_avg["Year"]==2022]

    # merge avg hours
    rel_df_no_orf_pt_hrs = rel_df_no_orf.merge(
        port_hours_avg_2022,
        left_on=["Calendary_Year", "Month(int)", "seaport_code"],
        right_on=["Year", "Month", "seaport_code"]
    )


    # RETAIL SALES

    # schedule + retail

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

    rel_df_nona.loc[:, "region"] = rel_df_nona["POL"].apply(
        lambda x: rel_port_map[x]
    )

    sales_df = sales_data

    # process retail sales data
    new_cols = [col.strip() for col in sales_df.columns]
    sales_df.columns = new_cols

    sales_df.loc[:, "month"] = sales_df["MonthYear"].apply(
        lambda x: int(x.split("/")[0])
    )

    sales_df.loc[:, "year"] = sales_df["MonthYear"].apply(
        lambda x: int(x.split("/")[1])
    )

    sales_df.loc[:, "date"] = sales_df["MonthYear"].apply(
        lambda x: datetime.datetime.strptime(
            x, "%m/%Y"
        )
    )

    # create offset date column
    sales_df.loc[:, "date(offset)"] = sales_df['date'] + pd.DateOffset(months=1)

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
                sales_df["date(offset)"],
                sales_df[region]
            )
        )

        date_region_sales[region] = region_dict


    # calculate max date to avoid index error
    max_date = sales_df["date(offset)"].max()

    # finally, create new columns
    # iterate over rows
    rel_df_nona.loc[:, "retail_sales"] = rel_df_nona.apply(
        lambda x: date_region_sales[x["region"]][x["Date"]] if x["Date"] <= max_date else None, axis=1
    )


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



    if partial_pred or overall_pred:
        # instantiate baseline model
        base_model = BaselineModel(train_df_filtered, label=label)
        preds = []
        preds_std = []
        print("Computing predictions...")  # TODO: progress bar
        for ind, row in val_X_filtered.iterrows():
            pred, pred_std = base_model.predict(*row)

            preds.append(pred)
            preds_std.append(pred_std)


        preds_array = np.array(preds)
        preds_std_array = np.array(preds_std)

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


    # linear regression
    # since we only have port call data up to august we restrict val_res
    if include_reg and overall_pred:
        val_res = val_res[val_res["Date"] < datetime.datetime(2022, 9, 1)]

        split_month = min(8, split_month)
        datetime_split = datetime.datetime(2022, split_month, 1)

        # linear regression split (port hours)
        train_X_rg, train_y_rg, val_X_rg, val_y_rg = get_reg_train_test(
            rel_df_no_orf_pt_hrs,
            datetime_split,
            label=label
        )

        # linear regression split (retail)
        train_X_rg_ret, train_y_rg_ret, val_X_rg_ret, val_y_rg_ret = get_reg_train_test(
            rel_df_nona, #rel_df_sales,
            datetime_split,
            label=label,
            use_retail=True
        )

        try:
            # evaluate linear regression
            linreg = LinearRegression()
            val_mae_rg, val_mape_rg, val_mae_over_rg, val_mape_over_rg, result_df_rg = compute_train_val_mae(
                linreg,
                train_X_rg,
                val_X_rg,
                train_y_rg,
                val_y_rg,
                calc_mape=True,
                label=label
            )

            # linreg = LinearRegression()  # I am no too sure if we need to instantiate twice
            val_mae_rg_ret, val_mape_rg_ret, val_mae_over_rg_ret, val_mape_over_rg_ret, result_df_rg_ret = compute_train_val_mae(
                linreg,
                train_X_rg_ret,
                val_X_rg_ret,
                train_y_rg_ret,
                val_y_rg_ret,
                calc_mape=True,
                label=label
            )

            eval_lin_reg = True

        except:
            raise Exception("Not enough data. Choose a different split")
