"""Run the whole lab3 pipeline in order."""

from __future__ import annotations

from task1_prepare_image_data import main as prepare_image_data
from task2_prepare_timeseries_data import main as prepare_timeseries_data
from task3_image_experiments import main as run_image_experiments
from task4_timeseries_experiments import main as run_timeseries_experiments


def main() -> None:
    prepare_image_data()
    prepare_timeseries_data()
    run_image_experiments()
    run_timeseries_experiments()


if __name__ == "__main__":
    main()
