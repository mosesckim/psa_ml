import os
import tqdm
import argparse, configparser

import pandas as pd

from bl_rec.src.utils import get_preds


def main():
    """Script main method to generate predictions on a random split"""
    parser = argparse.ArgumentParser()
    # add args
    parser.add_argument("-c", "--config_file_path", type=str, help='Config file')
    parser.add_argument("-p", "--path_to_data", default="bl_rec/data")
    parser.add_argument("-t", "--test_size", default=0.1)
    parser.add_argument("--replace_nulls", default=False)
    parser.add_argument("-f", "--csv_filename", default="ALLML2022.csv")
    # lane
    parser.add_argument("-n", "--necessary_fields", default=[])
    # target fields
    parser.add_argument("-u", "--unfilled_fields", default=[])

    # parse
    args = parser.parse_args()
    path_to_data = args.path_to_data
    test_size = args.test_size
    bdp_file = args.csv_filename
    replace_nulls = args.replace_nulls

    # parse config file and overwrite
    if args.config_file_path:
        config = configparser.ConfigParser()
        config.read(args.config_file_path)
        defaults = {}
        defaults.update(dict(config.items("lane")))
        parser.set_defaults(**defaults)
        args = parser.parse_args() # overwrite arguments


    # lane and target fields
    necessary_fields = eval(args.necessary_fields)
    unfilled_fields = eval(args.unfilled_fields)

    # read in file
    print("Reading input csv file...")

    bdp_file_path = os.path.join(path_to_data, bdp_file)

    if bdp_file == "ALLML2022.csv":
        bl_df = pd.read_csv(bdp_file_path, on_bad_lines="skip")
    else:
        bl_df = pd.read_excel(bdp_file_path, sheet_name="BDP December cleaned V2")

    res = {}

    for uf_field in tqdm.tqdm(unfilled_fields):
        stats = get_preds(
            bl_df,
            ff_fields=necessary_fields,
            uf_fields=[uf_field],
            test_size=test_size,
            replace_nulls=replace_nulls,
        )[1]

        acc = stats.accuracy
        cvg = stats["train coverage in val"]  # stats.coverage
        input_cvg = stats["train coverage in val (input)"]
        null_perc = stats.null_perc  # this only is meaningful if we don't replace nulls

        res[uf_field] = (acc, cvg, input_cvg, null_perc)

    res_df = pd.Series(res).to_frame()

    res_df.loc[:, "accuracy"] = res_df.apply(lambda x: x[0][0], axis=1)
    res_df.loc[:, "train coverage in val"] = res_df.apply(lambda x: x[0][1], axis=1)
    res_df.loc[:, "train coverage in val (input)"] = res_df.apply(
        lambda x: x[0][2], axis=1
    )
    res_df.loc[:, "null_perc"] = res_df.apply(lambda x: x[0][3], axis=1)

    res_df = res_df[
        [
            "accuracy",
            "train coverage in val",
            "train coverage in val (input)",
            "null_perc",
        ]
    ]

    res_df.columns = ["acc", "train cvg", "train cvg (input)", "null_perc"]

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save result
    output_filename = "result_lane_cvg.csv"
    res_df.to_csv(os.path.join(output_dir, output_filename))


if __name__ == "__main__":
    main()
