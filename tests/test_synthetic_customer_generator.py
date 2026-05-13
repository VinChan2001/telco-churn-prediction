import unittest

from src.synthetic_customer_generator import FEATURE_COLUMNS, generate_customers


INTERNET_DEPENDENT_COLUMNS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


class SyntheticCustomerGeneratorTest(unittest.TestCase):
    def test_generated_customers_match_feature_contract(self):
        customers = generate_customers(n_customers=250, seed=123)

        self.assertEqual(customers.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(len(customers), 250)

    def test_generated_customers_respect_service_constraints(self):
        customers = generate_customers(n_customers=500, seed=456)

        no_phone = customers["Phone Service"] == "No"
        self.assertTrue(
            (
                customers.loc[no_phone, "Multiple Lines"]
                == "No phone service"
            ).all()
        )

        no_internet = customers["Internet Service"] == "No"
        for column in INTERNET_DEPENDENT_COLUMNS:
            self.assertTrue(
                (
                    customers.loc[no_internet, column]
                    == "No internet service"
                ).all()
            )

    def test_total_charges_follow_monthly_charges_and_tenure(self):
        customers = generate_customers(n_customers=250, seed=789)
        expected_total = (
            customers["Monthly Charges"] * customers["Tenure Months"]
        ).round(2)

        self.assertTrue(
            (customers["Total Charges"].round(2) == expected_total).all()
        )


if __name__ == "__main__":
    unittest.main()
