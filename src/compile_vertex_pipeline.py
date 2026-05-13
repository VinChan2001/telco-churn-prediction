import argparse
from pathlib import Path

from kfp import compiler

from src.vertex_champion_pipeline import vertex_champion_challenger_pipeline


DEFAULT_OUTPUT_PATH = "pipelines/vertex_champion_challenger_pipeline.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the Vertex AI champion/challenger pipeline."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compiler.Compiler().compile(
        pipeline_func=vertex_champion_challenger_pipeline,
        package_path=str(output_path),
    )

    print(f"Compiled pipeline: {output_path}")


if __name__ == "__main__":
    main()
