try:
    from src.config import SYNTHETIC_CUSTOMERS_PATH
    from src.synthetic_customer_generator import generate_customers
except ModuleNotFoundError:
    from config import SYNTHETIC_CUSTOMERS_PATH
    from synthetic_customer_generator import generate_customers


def main():
    """Generate and save synthetic customer records."""

    print("Generating synthetic customer data...")

    synthetic_df = generate_customers(n_customers=1000, seed=42)

    print("Saving synthetic customers...")
    synthetic_df.to_csv(SYNTHETIC_CUSTOMERS_PATH, index=False)

    print("Synthetic customer generation complete.")
    print(f"Saved to: {SYNTHETIC_CUSTOMERS_PATH}")
    print(f"Rows generated: {len(synthetic_df)}")
    print(f"Columns generated: {len(synthetic_df.columns)}")


if __name__ == "__main__":
    main()
