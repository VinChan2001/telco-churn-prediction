from src.bq_model_input import MODEL_INPUT_TABLE_ID, refresh_model_input_table


def main() -> None:
    table = refresh_model_input_table()

    print(f"Refreshed {MODEL_INPUT_TABLE_ID}")
    print(f"Rows available for inference: {table.num_rows}")


if __name__ == "__main__":
    main()
