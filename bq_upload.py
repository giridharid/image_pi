"""
Acquink Solutions  |  GS1 DataKart BigQuery Upload
===================================================
Creates the BigQuery table and loads merged product data.

Run AFTER merge_pipeline.py:
  python bq_upload.py --csv ./output_merged/gs1_products_merged.csv

Author: Acquink Solutions | acquink.com
"""

from __future__ import annotations
import argparse, json, warnings
from pathlib import Path

PROJECT  = "gen-lang-client-0143536012"
DATASET  = "gs1_datakart"
TABLE    = "products"
LOCATION = "asia-south1"

SCHEMA_JSON = [
    {"name": "barcode",               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "brand",                 "type": "STRING",  "mode": "NULLABLE"},
    {"name": "product_name",          "type": "STRING",  "mode": "NULLABLE"},
    {"name": "product_type",          "type": "STRING",  "mode": "NULLABLE"},
    {"name": "fssai",                 "type": "STRING",  "mode": "NULLABLE"},
    {"name": "mrp",                   "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "net_weight",            "type": "STRING",  "mode": "NULLABLE"},
    {"name": "best_before",           "type": "STRING",  "mode": "NULLABLE"},
    {"name": "batch_no",              "type": "STRING",  "mode": "NULLABLE"},
    {"name": "manufacturer_name",     "type": "STRING",  "mode": "NULLABLE"},
    {"name": "manufacturer_address",  "type": "STRING",  "mode": "NULLABLE"},
    {"name": "complaint_number",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "complaint_email",       "type": "STRING",  "mode": "NULLABLE"},
    {"name": "website",               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "country_of_origin",     "type": "STRING",  "mode": "NULLABLE"},
    {"name": "certifications",        "type": "STRING",  "mode": "NULLABLE"},
    {"name": "key_claims",            "type": "STRING",  "mode": "NULLABLE"},
    {"name": "ingredients",           "type": "STRING",  "mode": "NULLABLE"},
    {"name": "ingredient_count",      "type": "INT64",   "mode": "NULLABLE"},
    {"name": "languages",             "type": "STRING",  "mode": "NULLABLE"},
    {"name": "language_codes",        "type": "STRING",  "mode": "NULLABLE"},
    {"name": "language_count",        "type": "INT64",   "mode": "NULLABLE"},
    {"name": "regions",               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "region_codes",          "type": "STRING",  "mode": "NULLABLE"},
    {"name": "storage",               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "panels_processed",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "confidence",            "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "flags",                 "type": "STRING",  "mode": "NULLABLE"},
    # Nutrients
    {"name": "n_energy",              "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_energy_unit",         "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_protein",             "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_protein_unit",        "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_total_carbohydrate",  "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_total_carbohydrate_unit","type": "STRING","mode": "NULLABLE"},
    {"name": "n_total_sugars",        "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_total_sugars_unit",   "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_added_sugars",        "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_added_sugars_unit",   "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_total_fat",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_total_fat_unit",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_saturated_fat",       "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_saturated_fat_unit",  "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_trans_fat",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_trans_fat_unit",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_cholesterol",         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_cholesterol_unit",    "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_sodium",              "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_sodium_unit",         "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_dietary_fiber",       "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_dietary_fiber_unit",  "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_vitamin_a",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_vitamin_a_unit",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_vitamin_c",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_vitamin_c_unit",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_vitamin_d",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_vitamin_d_unit",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_calcium",             "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_calcium_unit",        "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_iron",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_iron_unit",           "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_potassium",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_potassium_unit",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_zinc",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_zinc_unit",           "type": "STRING",  "mode": "NULLABLE"},
    {"name": "n_magnesium",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "n_magnesium_unit",      "type": "STRING",  "mode": "NULLABLE"},
]


def ensure_dataset(client):
    from google.cloud import bigquery
    try:
        client.get_dataset(DATASET)
        print(f"  Dataset {DATASET}: exists")
    except Exception:
        ds = bigquery.Dataset(f"{PROJECT}.{DATASET}")
        ds.location = LOCATION
        client.create_dataset(ds)
        print(f"  Dataset {DATASET}: created")


def ensure_table(client):
    from google.cloud import bigquery
    table_ref = f"{PROJECT}.{DATASET}.{TABLE}"
    try:
        client.get_table(table_ref)
        print(f"  Table {TABLE}: exists")
    except Exception:
        schema = [bigquery.SchemaField(
            f["name"], f["type"], mode=f["mode"]) for f in SCHEMA_JSON]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        print(f"  Table {TABLE}: created")
    return table_ref


def upload_csv(csv_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT, location=LOCATION)

    print(f"\nUploading {csv_path} to BigQuery...")
    ensure_dataset(client)
    table_ref = ensure_table(client)

    job_config = bigquery.LoadJobConfig(
        source_format    = bigquery.SourceFormat.CSV,
        skip_leading_rows= 1,
        autodetect       = False,
        schema           = [bigquery.SchemaField(
            f["name"], f["type"], mode=f["mode"]) for f in SCHEMA_JSON],
        write_disposition= bigquery.WriteDisposition.WRITE_TRUNCATE,
        allow_quoted_newlines = True,
    )

    with open(csv_path, "rb") as f:
        job = client.load_table_from_file(f, table_ref, job_config=job_config)

    print("  Loading... ", end="", flush=True)
    job.result()
    print("Done.")

    table = client.get_table(table_ref)
    print(f"\n  Rows loaded  : {table.num_rows}")
    print(f"  Table        : {PROJECT}.{DATASET}.{TABLE}")
    print(f"  Location     : {LOCATION}")
    print(f"\n  BigQuery URL : https://console.cloud.google.com/bigquery"
          f"?project={PROJECT}&ws=!1m5!1m4!4m3!1s{PROJECT}!2s{DATASET}!3s{TABLE}")


def main():
    parser = argparse.ArgumentParser(description="Upload GS1 merged CSV to BigQuery")
    parser.add_argument("--csv", required=True, help="Path to gs1_products_merged.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    upload_csv(csv_path)


if __name__ == "__main__":
    main()
