import argparse

from .inferencer import Inferencer
from .utils import Model_Args, Project_Args


def run(
    project_args: Project_Args,
    model_args: Model_Args,
) -> None:
    inferencer = Inferencer(project_args._asdict(), **model_args._asdict())
    inferencer()


def main():
    parser = argparse.ArgumentParser(description="Process a zip file.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to axsy-notation-data bucket",
    )
    parser.add_argument(
        "--dataset_id", type=str, default="dataset-1", help="Dataset/project name"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input image or directory name. Defaults to whole images directory",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo.pt",
        help="Name of model to use for label assist. Must be in label-assist directory",
    )
    parser.add_argument("--no_inference", action="store_true", help="Disable inference")

    args = parser.parse_args()

    project_args = Project_Args(
        path=args.path,
        dataset_id=args.dataset_id,
        input=args.input,
    )

    model_args = Model_Args(
        detector=args.detector,
        run_inference=not (args.no_inference or args.detector is None),
    )

    run(project_args, model_args)


if __name__ == "__main__":
    import warnings

    warnings.warn(
        "use 'python -m axsy_inference', not 'python -m axsy_inference.run'",
        DeprecationWarning,
    )
    main()
