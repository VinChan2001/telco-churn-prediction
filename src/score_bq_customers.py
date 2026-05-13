from src.bq_scorer import score_unscored_customers


def main() -> None:
    result = score_unscored_customers(refresh_input=True)

    print(f"Rows scored: {result.rows_scored}")
    print(f"Files scored: {result.files_scored}")
    print(f"Scored customers table: {result.scored_customers_table}")
    print(f"Scored files table: {result.scored_files_table}")
    print(f"Threshold: {result.threshold}")


if __name__ == "__main__":
    main()
