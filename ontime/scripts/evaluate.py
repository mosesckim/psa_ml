# generic evaluation script
# given model
# trade lane
# and/or set of routes

import yaml
import os

import pandas as pd
import numpy as np


from ontime.src.models import Baseline


if __name__ == "__main__":

    with open("ontime/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # data dir
    data_dir = config["data_path"]

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
