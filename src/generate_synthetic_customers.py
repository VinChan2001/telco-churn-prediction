import random

import pandas as pd
from faker import Faker

from config import SYNTHETIC_CUSTOMERS_PATH


fake = Faker()
random.seed(42)
Faker.seed(42)


FEATURE_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]


def generate_customer():
    """Generate one synthetic Telco-like customer record."""

    gender = random.choice(["Male", "Female"])
    senior_citizen = random.choice(["Yes", "No"])
    partner = random.choice(["Yes", "No"])
    dependents = random.choice(["Yes", "No"])

    tenure_months = random.randint(0, 72)

    phone_service = random.choice(["Yes", "No"])

    if phone_service == "No":
        multiple_lines = "No phone service"
    else:
        multiple_lines = random.choice(["Yes", "No"])

    internet_service = random.choice(["DSL", "Fiber optic", "No"])

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        monthly_charges = round(random.uniform(18, 35), 2)
    else:
        online_security = random.choice(["Yes", "No"])
        online_backup = random.choice(["Yes", "No"])
        device_protection = random.choice(["Yes", "No"])
        tech_support = random.choice(["Yes", "No"])
        streaming_tv = random.choice(["Yes", "No"])
        streaming_movies = random.choice(["Yes", "No"])

        if internet_service == "Fiber optic":
            monthly_charges = round(random.uniform(70, 120), 2)
        else:
            monthly_charges = round(random.uniform(35, 85), 2)

    contract = random.choice(["Month-to-month", "One year", "Two year"])
    paperless_billing = random.choice(["Yes", "No"])

    payment_method = random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )

    total_charges = round(monthly_charges * tenure_months, 2)

    customer = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure_months,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
    }

    return customer


def generate_customers(n_customers=1000):
    """Generate a synthetic batch of new customers."""

    customers = [generate_customer() for _ in range(n_customers)]

    synthetic_df = pd.DataFrame(customers)

    synthetic_df = synthetic_df[FEATURE_COLUMNS]

    return synthetic_df


def main():
    """Generate and save synthetic customer records."""

    print("Generating synthetic customer data...")

    synthetic_df = generate_customers(n_customers=1000)

    print("Saving synthetic customers...")
    synthetic_df.to_csv(SYNTHETIC_CUSTOMERS_PATH, index=False)

    print("Synthetic customer generation complete.")
    print(f"Saved to: {SYNTHETIC_CUSTOMERS_PATH}")
    print(f"Rows generated: {len(synthetic_df)}")
    print(f"Columns generated: {len(synthetic_df.columns)}")


if __name__ == "__main__":
    main()