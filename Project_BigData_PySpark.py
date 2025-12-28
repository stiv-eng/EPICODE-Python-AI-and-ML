# Progetto Finale : BigData e PySpark
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import pyarrow as pa
import pyarrow.orc as orc

# Parte 1: 
for chunk in pd.read_csv("Orders_Azienda.csv", chunksize = 10):

    # Applica tecniche di ottimizzazione: 

    # Modifica tipo di dato: 
    chunk["region"] = chunk["region"].astype("category")
    chunk["sales_rep"] = chunk["sales_rep"].astype("category")
    chunk["product_category"] = chunk["product_category"].astype("category")
    chunk["customer_type"] = chunk["customer_type"].astype("category")
    chunk["order_id"] = chunk["order_id"].astype("int16")
    chunk["units_sold"] = chunk["units_sold"].astype("int16")

# Parte 2: Introduci Dask

def main():
    client = Client()

    # Completa operazioni su Dask: 
    ddf = dd.read_csv("Orders_Azienda")
    somma_revenue = ddf["revenue"].sum().compute()
    print(f"Somma totale {somma_revenue}$")

    # Usa diversi scheduler per completare operazioni: 

    tot_units_sold = ddf["units_sold"].sum().compute(scheduler = "threads")
    tot_units_sold_2 = ddf["units_sold"].sum().compute(scheduler = "processes")

    # Applica operazioni distribuite:
    vendite_per_rep = ddf.groupby("sales_rep")["revenue"].sum()
    print("Vendite di ogni Representative: \n", vendite_per_rep)

    client.close()

if __name__ == "__main__":
    main()


# Parte 3: Architettura Big Data e storage distribuito 
table = pa.Table.from_pandas()

