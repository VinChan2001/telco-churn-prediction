import unittest

import pandas as pd

from src.synthetic_customer_generator import FEATURE_COLUMNS
from src.training_trigger import (
    TARGET_COLUMN,
    build_training_run_id,
    is_in_training_prefix,
    is_supported_training_file,
    validate_training_dataframe,
)


class TrainingTriggerTest(unittest.TestCase):
    def test_build_training_run_id_includes_generation_identity(self):
        gcs_uri = "gs://bucket/training/incoming/churn.csv"

        first_run_id = build_training_run_id(gcs_uri, "1")
        second_run_id = build_training_run_id(gcs_uri, "2")

        self.assertNotEqual(first_run_id[-8:], second_run_id[-8:])

    def test_training_file_filters_expected_prefix_and_formats(self):
        self.assertTrue(is_in_training_prefix("training/incoming/churn.xlsx"))
        self.assertFalse(is_in_training_prefix("synthetic/customers.csv"))
        self.assertTrue(is_supported_training_file("training/incoming/churn.csv"))
        self.assertTrue(is_supported_training_file("training/incoming/churn.xlsx"))
        self.assertFalse(is_supported_training_file("training/incoming/readme.txt"))

    def test_validate_training_dataframe_requires_model_contract(self):
        row = {column: "Yes" for column in FEATURE_COLUMNS}
        row.update(
            {
                "Tenure Months": 12,
                "Monthly Charges": 80.0,
                "Total Charges": 960.0,
                TARGET_COLUMN: 1,
            }
        )
        dataframe = pd.DataFrame([row] * 2)

        validation = validate_training_dataframe(dataframe, minimum_row_count=2)

        self.assertEqual(validation.row_count, 2)
        self.assertEqual(validation.missing_columns, [])

    def test_validate_training_dataframe_rejects_missing_target(self):
        dataframe = pd.DataFrame([{column: "Yes" for column in FEATURE_COLUMNS}])

        with self.assertRaisesRegex(ValueError, TARGET_COLUMN):
            validate_training_dataframe(dataframe, minimum_row_count=1)

    def test_validate_training_dataframe_rejects_bad_target_values(self):
        row = {column: "Yes" for column in FEATURE_COLUMNS}
        row[TARGET_COLUMN] = "Yes"
        dataframe = pd.DataFrame([row])

        with self.assertRaisesRegex(ValueError, "0/1"):
            validate_training_dataframe(dataframe, minimum_row_count=1)


if __name__ == "__main__":
    unittest.main()
