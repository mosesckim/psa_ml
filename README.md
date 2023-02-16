# psa_ml

## Environment

Create a new environment using `virtualenv` or `conda`. Make sure `pip` is pointing to the correct path (i.e. inside your environment) by running `which pip`.

## Install package (CLI)

Once you have created your environment activate it and `cd` into the project repository root and run:

    pip install .


The command above will install necessary dependencies and packages found inside `bl_rec/` and `ontime/`. Once the subpackages are installed, we can run the BL rec and ontime scripts straight from the command line as shown below:

### BL

    > bl_get_preds -h

    usage: bl_get_preds [-h] [-c CONFIG_FILE_PATH] [-p PATH_TO_DATA] [-t TEST_SIZE]
                    [--replace_nulls REPLACE_NULLS] [-f CSV_FILENAME] [-n NECESSARY_FIELDS]
                    [-u UNFILLED_FIELDS]

    optional arguments:
    -h, --help            show this help message and exit
    -c CONFIG_FILE_PATH, --config_file_path CONFIG_FILE_PATH
                            Config file
    -p PATH_TO_DATA, --path_to_data PATH_TO_DATA
    -t TEST_SIZE, --test_size TEST_SIZE
    --replace_nulls REPLACE_NULLS
    -f CSV_FILENAME, --csv_filename CSV_FILENAME
    -n NECESSARY_FIELDS, --necessary_fields NECESSARY_FIELDS
    -u UNFILLED_FIELDS, --unfilled_fields UNFILLED_FIELDS

Before using the CLI above, make sure to store either `ALLML2022.csv` or `BDP cleaned full Plant Code.xlsx` in `bl_rec/data/`. Then, from the project root, run

    bl_get_preds -c bl_rec/configs/lane.conf

in your terminal (command will use the default script params). However, if you wish to run this script from a different path make sure to alter the corresponding params (i.e. `PATH_TO_DATA` and `CSV_FILENAME`) accordingly. To change default lane fields and target fields, please access the corresponding config file at `bl_rec/configs/lane.conf`.


### ONTIME

    > ontime_evaluate -h

    usage: ontime_evaluate [-h] [-d DATA_DIR_PATH] [--split_month SPLIT_MONTH] [--max_month MAX_MONTH]
                       [--label LABEL] [--partial_pred PARTIAL_PRED] [--overall_pred OVERALL_PRED]
                       [--restrict_trade RESTRICT_TRADE] [--trade_option TRADE_OPTION]
                       [--carrier_option CARRIER_OPTION] [--service_option SERVICE_OPTION]
                       [--eval_lin_reg EVAL_LIN_REG] [--include_reg INCLUDE_REG]
                       [--schedule_filename SCHEDULE_FILENAME]

    optional arguments:
    -h, --help            show this help message and exit
    -d DATA_DIR_PATH, --data_dir_path DATA_DIR_PATH
    --split_month SPLIT_MONTH
    --max_month MAX_MONTH
    --label LABEL
    --partial_pred PARTIAL_PRED
    --overall_pred OVERALL_PRED

Similarly, place files `2022-11-29TableMapping_Reliability-SCHEDULE_RELIABILITY_PP.csv`, `retail Sales 202210.xlsx`, and `CPI Core 202210.xlsx` in the `ontime/data/` directory and run

    ontime_evaluate

