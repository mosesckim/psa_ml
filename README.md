# psa_ml

## Environment

Create a new environment using `virtualenv` or `conda`. Make sure `pip` is pointing to the correct path (i.e. inside your environment) by running `which pip`.

## Install package (CLI)

Once you have created your environment activate it and `cd` into the project repository root and run:

    pip install .


The command above will install necessary dependencies and packages found inside `bl_rec/` and `ontime/`. Once the subpackages are installed, we can run the BL rec and ontime scripts straight from the command line as shown below:

### BL

    > bl_get_preds -h

    usage: bl_get_preds [-h] [-p PATH_TO_DATA] [-t TEST_SIZE] [--replace_nulls REPLACE_NULLS]
                    [-f CSV_FILENAME]

    optional arguments:
    -h, --help            show this help message and exit
    -p PATH_TO_DATA, --path_to_data PATH_TO_DATA
    -t TEST_SIZE, --test_size TEST_SIZE
    --replace_nulls REPLACE_NULLS
    -f CSV_FILENAME, --csv_filename CSV_FILENAME

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