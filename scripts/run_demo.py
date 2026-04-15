import argparse
from pathlib import Path
from highway_perception.pipeline import HighwayPerceptionPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models-cfg", default="configs/models.yaml")
    p.add_argument("--thresholds-cfg", default="configs/thresholds.yaml")
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pipeline = HighwayPerceptionPipeline.from_files(args.models_cfg, args.thresholds_cfg)
    print("Demo complete:", pipeline.run_video(args.video, args.output))


if __name__ == "__main__":
    main()
