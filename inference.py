import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from utils.tools import get_args
from runner import str2trainer

# checkpoint_path = 'save/rebuttal_dataset_7/'
checkpoint_path = 'save/rebuttal_0.5_0_0.01_epsilon_1/'

def get_runner(args):
    runner = str2trainer[args.runner](args, inference=True)
    for i in range(8, 9):
        print(i)
        runner.load(save_dir=checkpoint_path, epoch=i)
        # runner.evaluate(split="test")
        runner.evaluate(split="test", save_dir=args.save_dir, epoch=i)



if __name__ == "__main__":
    import pickle

    args_filename = "args-{}.pkl".format(0)
    with open(os.path.join(checkpoint_path, args_filename), "rb") as f:
        args = pickle.load(f)
    args.batch_size = 120
    runner = get_runner(args)
