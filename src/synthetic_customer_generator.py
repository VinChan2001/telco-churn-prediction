import random

import pandas as pd


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


def generate_customer(rng: random.Random) -> dict:
    """Generate one synthetic Telco-like customer record."""

    gender = rng.choice(["Male", "Female"])
    senior_citizen = rng.choice(["Yes", "No"])
    partner = rng.choice(["Yes", "No"])
    dependents = rng.choice(["Yes", "No"])

    tenure_months = rng.randint(0, 72)

    phone_service = rng.choice(["Yes", "No"])

    if phone_service == "No":
        multiple_lines = "No phone service"
    else:
        multiple_lines = rng.choice(["Yes", "No"])

    internet_service = rng.choice(["DSL", "Fiber optic", "No"])

    if internet_service == "No":
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        monthly_charges = round(rng.uniform(18, 35), 2)
    else:
        online_security = rng.choice(["Yes", "No"])
        online_backup = rng.choice(["Yes", "No"])
        device_protection = rng.choice(["Yes", "No"])
        tech_support = rng.choice(["Yes", "No"])
        streaming_tv = rng.choice(["Yes", "No"])
        streaming_movies = rng.choice(["Yes", "No"])

        if internet_service == "Fiber optic":
            monthly_charges = round(rng.uniform(70, 120), 2)
        else:
            monthly_charges = round(rng.uniform(35, 85), 2)

    contract = rng.choice(["Month-to-month", "One year", "Two year"])
    paperless_billing = rng.choice(["Yes", "No"])

    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )

    total_charges = round(monthly_charges * tenure_months, 2)

    return {
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


def generate_customers(n_customers: int = 1000, seed: int | None = None) -> pd.DataFrame:
    """Generate a synthetic batch using the model's raw feature contract."""

    rng = random.Random(seed)
    customers = [generate_customer(rng) for _ in range(n_customers)]
    synthetic_df = pd.DataFrame(customers)

    return synthetic_df[FEATURE_COLUMNS]
