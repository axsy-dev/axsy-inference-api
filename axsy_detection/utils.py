from typing import NamedTuple


class Project_Args(NamedTuple):
    path: str
    dataset_id: str
    input: str


class Model_Args(NamedTuple):
    detector: str
    run_inference: bool
