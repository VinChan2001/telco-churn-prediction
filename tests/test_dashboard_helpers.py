import unittest

import pandas as pd

from app.streamlit_app import (
    dashboard_summary,
    format_currency,
    format_int,
    format_percent,
    format_timestamp,
    normalize_scored_customers,
    pipeline_is_in_sync,
)


class DashboardHelpersTest(unittest.TestCase):
    def test_formatters(self):
        self.assertEqual(format_int(1200), "1,200")
        self.assertEqual(format_currency(12345.67), "$12,346")
        self.assertEqual(format_percent(0.1234), "12.3%")
        self.assertEqual(
            format_timestamp("2026-05-13T18:01:09Z"),
            "2026-05-13 18:01 UTC",
        )

    def test_normalize_scored_customers_adds_missing_fields(self):
        customers = pd.DataFrame(
            [
                {
                    "Monthly Charges": 100.0,
                    "churn_probability": "0.75",
                    "predicted_churn": "1",
                },
                {
                    "Monthly Charges": 50.0,
                    "churn_probability": "0.20",
                    "predicted_churn": "0",
                },
            ]
        )

        normalized = normalize_scored_customers(customers)

        self.assertEqual(normalized["predicted_churn"].tolist(), [1, 0])
        self.assertEqual(
            normalized["estimated_annual_revenue"].tolist(),
            [1200.0, 600.0],
        )
        self.assertEqual(
            normalized["risk_segment"].tolist(),
            ["High Risk", "Low Risk"],
        )

    def test_dashboard_summary_counts_key_segments(self):
        customers = normalize_scored_customers(
            pd.DataFrame(
                [
                    {
                        "Monthly Charges": 100.0,
                        "churn_probability": 0.75,
                        "predicted_churn": 1,
                    },
                    {
                        "Monthly Charges": 50.0,
                        "churn_probability": 0.20,
                        "predicted_churn": 0,
                    },
                ]
            )
        )

        summary = dashboard_summary(customers)

        self.assertEqual(summary["total_customers"], 2)
        self.assertEqual(summary["high_risk_customers"], 1)
        self.assertEqual(summary["flagged_customers"], 1)
        self.assertEqual(summary["flagged_annual_revenue"], 1200.0)

    def test_pipeline_sync_requires_matching_counts_and_clean_quality(self):
        freshness = {
            "raw_rows": 10,
            "model_input_rows": 10,
            "scored_rows": 10,
            "unscored_traceable_rows": 0,
            "invalid_service_combos": 0,
            "null_total_charges": 0,
        }

        self.assertTrue(pipeline_is_in_sync(freshness))

        freshness["scored_rows"] = 9
        self.assertFalse(pipeline_is_in_sync(freshness))


if __name__ == "__main__":
    unittest.main()
