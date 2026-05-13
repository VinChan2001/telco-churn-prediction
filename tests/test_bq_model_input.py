import unittest

from src.bq_model_input import (
    METADATA_COLUMN_NAMES,
    MODEL_INPUT_TABLE_ID,
    RAW_SYNTHETIC_TABLE_ID,
    build_model_input_sql,
)
from src.synthetic_customer_generator import FEATURE_COLUMNS


class BigQueryModelInputSqlTest(unittest.TestCase):
    def test_model_input_sql_targets_expected_tables(self):
        sql = build_model_input_sql()

        self.assertIn(f"CREATE OR REPLACE TABLE `{MODEL_INPUT_TABLE_ID}`", sql)
        self.assertIn(f"FROM `{RAW_SYNTHETIC_TABLE_ID}`", sql)

    def test_model_input_sql_projects_feature_and_metadata_columns(self):
        sql = build_model_input_sql("raw_table", "model_input_table")

        for column in FEATURE_COLUMNS:
            self.assertIn(f"AS `{column}`", sql)

        for column in METADATA_COLUMN_NAMES:
            self.assertIn(f"AS `{column}`", sql)

    def test_model_input_sql_normalizes_service_dependent_values(self):
        sql = build_model_input_sql("raw_table", "model_input_table")

        self.assertIn("No phone service", sql)
        self.assertIn("No internet service", sql)
        self.assertIn("SAFE_CAST(`Total Charges` AS FLOAT64)", sql)


if __name__ == "__main__":
    unittest.main()
