import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from utils.tools import get_args
from runner import str2trainer


def get_runner(args):
    runner = str2trainer[args.runner](args)
    # runner.load(save_dir=args.restore_from, epoch=args.restore_epoch)
    runner.train()


if __name__ == "__main__":
    args = get_args()
    runner = get_runner(args)
