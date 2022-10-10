from module.evaluate import str2metric
import argparse
import pickle


class Evaluator:
    def __init__(self):
        self._build_evaluator()
        pass

    def _build_evaluator(self):
        self.metric = [str2metric["cider"](df="corpus"),
                       str2metric["bleu"](n_grams=[1, 2, 3, 4]),
                       str2metric["meteor"](),
                       str2metric["rouge"](),
                       str2metric["spice"](),
                       str2metric["accuracy"](metrics=["precision", "recall", "F1", "accuracy"])]
        return

    def eval(self, pred_collect, gt_collect):
        """

        Parameters
        ----------
        pred_collect: list[str]
        gt_collect: list[str]

        Returns
        -------

        """

        score, _ = self.metric[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("Metric {}: {:.4f}".format(str(self.metric[0]), score))

        score, _ = self.metric[1].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("Metric {}: @1 - {:.4f}, @2 - {:.4f}, @3 - {:.4f}, @4 - {:.4f}".format(
            str(self.metric[1]), score[0], score[1], score[2], score[3]))

        score, _ = self.metric[2].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("Metric {}: {:.4f}".format(str(self.metric[2]), score))

        score, _ = self.metric[3].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("Metric {}: {:.4f}".format(str(self.metric[3]), score))

        score, _ = self.metric[4].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        print("Metric {}: {:.4f}".format(str(self.metric[4]), score))



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default='/home/shiina/shiina/question/iq/results_cocoqa/20-pred.pkl', type=str, help='path to the config file')
    parser.add_argument('--gt', default="/home/shiina/shiina/question/iq/results_cocoqa/20-gt.pkl", type=str, help='rank')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.pred, "rb") as f:
        pred = pickle.load(f)
    with open(args.gt, "rb") as f:
        gt = pickle.load(f)
    evaluator = Evaluator()
    evaluator.eval(pred_collect=pred, gt_collect=gt)
