import pandas as pd

from sklearn.model_selection import train_test_split
from bl_rec.src.prob_model import FFProbSpace


def get_input_tuples(df: pd.DataFrame, cols=[]):
    """Return list of tuples corresponding to dataframe column values

    Args:
        df (pd.DataFrame): dataframe of bl form entries
        cols (list, optional): list of column names. Defaults to [].

    Returns:
        list: list of tuples where each entry corresponds to a column value
    """


    input_tuples = list(
        zip(
            *[df[col] for col in cols]
        )
    )
    return input_tuples


def get_preds(
    df: pd.DataFrame,
    ff_fields=["PortofLoadID", "LetterofCreditClause", "PlantCode", "CONO"],
    uf_fields=["ExporterAddressCode"],
    test_size=0.1    # TODO: add to config params
):

    ffuf_eg_df = df[ff_fields + uf_fields]

    # instead of dropping NA rows we replace NAs with the string "null"
    # ffuf_eg_df_no_na = ffuf_eg_df.fillna("null")  # TODO: include as config param
    ffuf_eg_df_no_na = ffuf_eg_df.dropna()  # TODO: add relevant config param

    print("Train data size: ", ffuf_eg_df.shape)

    # compute total number of null rows
    total_no_rows = ffuf_eg_df.shape[0]
    total_field_null_rows = ffuf_eg_df.shape[0] - ffuf_eg_df_no_na.shape[0]

    # train, test split
    ffuf_train , ffuf_val = train_test_split(ffuf_eg_df_no_na, test_size=test_size)

    # TRAIN
    # build prob space on train data (with single target variable)

    # need to make sure train rows have no nulls
    # ffuf_train = ffuf_train[[ff_fields]]
    # print("ffuf_train rows with nulls: ", ffuf_train.shape[0])
    # ffuf_train.dropna(inplace=True)
    # print("ffuf_train rows without nulls: ", ffuf_train.shape[0])

    ff_prob_sp = FFProbSpace(ffuf_train, ff_fields)
    ff_prob_sp.compute_prob_sp(uf_fields[0])

    # PREDICT

    # COMMON TARGET VALUES
    # let's see how many values ExporterAddressCode train and val have in common
    target_train_unique = ffuf_train[uf_fields[0]].unique()
    target_val_unique = ffuf_val[uf_fields[0]].unique()
    target_train_val_inter = set(target_train_unique).intersection(set(target_val_unique))

    # so we restrict the validation set further
    # to include the intersection field values only
    ffuf_val_res = ffuf_val[ffuf_val[uf_fields[0]].isin(target_train_val_inter)]


    # COMMON INPUT VALUES
    train_inputs = get_input_tuples(ffuf_train, cols=ff_fields)
    val_inputs = get_input_tuples(ffuf_val_res, cols=ff_fields)
    train_inputs_set = set(train_inputs)
    val_inputs_set = set(val_inputs)

    train_val_inputs_inter = train_inputs_set.intersection(val_inputs_set)

    # TODO: find how many rows in val live inside train_val_inputs_inter

    # number of rows in this intersection
    input_tuple_ser = ffuf_eg_df_no_na.apply(
        lambda x: tuple((x[col] for col in ff_fields)),
        axis=1
    )

    mask = input_tuple_ser.isin(train_val_inputs_inter)
    total_train_test_rows = input_tuple_ser[mask].shape[0]

    # evaluate
    # iterate over common field values
    preds = []
    counts = []
    for input_tuple in train_val_inputs_inter:

        pred_ser = ff_prob_sp.compute_cond_prob(input_tuple)
        max_prob = pred_ser.max()  # find max prob


        max_pred_val = pred_ser[pred_ser==max_prob].index[0]  # find value corresp. to max prob
        # make sure output is a string if not cast it
        max_pred_val_str = str(max_pred_val)  # TODO: do the same on train data

        preds.append(max_pred_val_str)


    # PREDICTIONS
    cols_list = list(zip(*train_val_inputs_inter))

    pred_df = pd.DataFrame(
        dict(
            zip(
                ff_fields + ["pred"],
                cols_list + [preds]
            )
        )
    )

    # ACCURACY
    # merge dataframes
    result_df = ffuf_val_res.merge(pred_df, on=ff_fields)
    acc = (result_df[uf_fields[0]] == result_df["pred"]).mean()

    # SUMMARY STATS
    ser_stats_input = {
        "ff_vars": ff_fields,
        "uf_var": uf_fields[0],
        "total_no_rows": total_no_rows,
        "total_field_null_rows": total_field_null_rows,
        "target_train_unique": len(target_train_unique),
        "target_val_unique": len(target_val_unique),
        "target_train_val_inter": len(target_train_val_inter),
        "ffuf_train.shape": ffuf_train.shape,
        "ffuf_val.shape": ffuf_val.shape,
        "ffuf_val_res.shape": ffuf_val_res.shape,
        "train coverage in val": ffuf_val_res.shape[0] / ffuf_val.shape[0],
        "train coverage in val (input)": len(train_val_inputs_inter) / len(val_inputs_set),
        "len(train_inputs)": len(train_inputs),
        "len(val_inputs)": len(val_inputs),
        "train_inputs_unique": len(train_inputs_set),
        "val_inputs_unique": len(val_inputs_set),
        "train_val_inputs_inter": len(train_val_inputs_inter),
        "total_train_test_rows": total_train_test_rows,
        "not null val input percentage": total_train_test_rows / total_no_rows,
        "null_perc": total_field_null_rows / total_no_rows,
        "accuracy": acc,

    }

    ser_stats = pd.Series(ser_stats_input)

    return result_df, ser_stats
