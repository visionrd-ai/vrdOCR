import argparse
import os, datetime
from pathlib import Path

from utils.config import get_config
from model.registeries import build_model
from logger import set_logger, logger

from tasks.train import train

DEFAULT_SAVE_DIR =  (Path(__file__).resolve().parent / "output").as_posix()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognition model')
    parser.add_argument('--config', '-c', type=str, required=True, help='config file')
    parser.add_argument('-o',
                    '--override',
                    action='append',
                    default=[],
                    help='config options to be overridden')
    
    parser.add_argument('--save-path', '-s', type=str, default=DEFAULT_SAVE_DIR, help='where to save the model')
    parser.add_argument("--run-name", type=str, default="default", help="name of this run")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test", action="store_true")
    group.add_argument("--evaluation", action="store_true")
    group.add_argument("--inference", action="store_true")
    parser.set_defaults(mode="train")

    parser.add_argument('--weights', '-w', type=str, help='model weights')
    parser.add_argument('--vis', action='store_true', help='whether to visualize results')
    parser.add_argument('--input', '-i', type=str, help='input image or folder')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Loading model config...")
    config = get_config(args.config, args.override)
    
    mode = (
        "test" if args.test else
        "eval" if args.evaluation else
        "inference" if args.inference else
        "train"
    )
    
    print("Building model using Registry...")
    model = build_model(config).cuda()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = f"{args.run_name}_{timestamp}"
    else:
        run_name = timestamp
    run_dir = f"{DEFAULT_SAVE_DIR}/{mode}_{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    
    logger = set_logger(f"{run_dir}/log.txt")
    logger.info(f"Run directory: {run_dir}")
    
    if mode == "train":
        print("Starting training...")
        train(
            config, 
            model, 
            Path(run_dir), 
            run_name, 
            logger=logger,
            vis=args.vis,
        )
        
    elif mode == "test":
        print("Starting testing...")
        
    elif mode == "eval":
        print("Starting evaluation...")

    else:
        print("Starting inference...")


if __name__ == '__main__':
    main()

