import random
import torch
import torch.nn as nn
import torch.nn.functional
from torch.serialization import save
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from model import discriminator
from utils.tools import get_log
from data.vqa2.data_loader import VQA2Dataset
from model import str2model
import time
import numpy as np
from module.evaluate import str2metric
import torch.distributed as dist
import torch.utils.data.distributed
import pickle, json
from utils.calculate import bbox_overlaps_batch
import math
from torch.utils.tensorboard import SummaryWriter
import shutil

from model.discriminator import Discriminator
from model.generator_vh_sample import Graph2seqGeneratorVhSamples
from utils.gan_utils import extract_visual_hint_from_prob
from module.submodule.loss import VisualHintLossFocal, VisualHintLossBalanced, VisualHintLossBalancedALL

from utils.gan_utils import pg_loss, vh_pg_loss
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from utils.tools import Scheduler, set_lr


class TrainerSeqGANVHSample:
    def __init__(self, args, inference=False):
        super(TrainerSeqGANVHSample, self).__init__()
        self.verbase = 1
        self.opt = args
        print("*********[Trainer configure]***********")
        print(self.opt)
        # os.environ["MASTER_PORT"] = "25076"
        
        self.accelerator = Accelerator(split_batches=True, fp16=True, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

        self.inference = inference
        self.__clean()
        self.__build_device()
        self.__build_logger(args.log_path)
        self.__build_dataloader(args)
        self.__build_model(args)

        self.__build_optimizer(args)
        self.__build_evaluator(args)
        # self.loss_vh = VisualHintLossBalancedALL()
        # self.scheduler = Scheduler(patience=self.opt.patience, delta=0, start_from=self.opt.lr_decay_epoch_start, trace_func=self.logger.info)
        self.loss_vh = VisualHintLossFocal(alpha=4, gamma=2)
    
    def __clean(self):
        if self.opt.clean and self.accelerator.is_local_main_process:
            print("-------")
            exit(0)
            if os.path.exists(self.opt.checkpoint_path):
                shutil.rmtree(self.opt.checkpoint_path)
            log_path = os.path.join(self.opt.log_path, self.opt.name)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
            if os.path.exists(self.opt.save_dir):
                shutil.rmtree(self.opt.save_dir)
        self.accelerator.wait_for_everyone()

    def __build_device(self):
        seed = int(self.opt.seed)
        random.seed(seed)
        np.random.seed(seed)
        if self.opt.use_gpu and torch.cuda.is_available():
            print('[ Using CUDA ]')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn
            cudnn.benchmark = True
            device = self.accelerator.device
        else:
            print('[ Using CPU ]')
            device = torch.device('cpu')
        self.device = device

    def __build_logger(self, log_path):
        log_path = os.path.join(log_path, self.opt.name)
        logger_path = os.path.join(log_path, "txt")
        tensorboard_path = os.path.join(log_path, "tensorboard")
        if not os.path.exists(logger_path) and self.accelerator.is_local_main_process:
            os.makedirs(logger_path)
        self.accelerator.wait_for_everyone()
        if not os.path.exists(tensorboard_path) and self.accelerator.is_local_main_process:
            os.makedirs(tensorboard_path)
        self.accelerator.wait_for_everyone()
        self.logger = get_log(os.path.join(logger_path, "log.txt"))
        self.writer = SummaryWriter(log_dir=tensorboard_path)

    def __build_dataloader(self, args):
        # train dataloader
        train_dataset = VQA2Dataset(split_dic_path=args.train_split_dic_path, vocab_path=args.vocab_path,
                                    split="train", ratio=args.train_set_ratio,
                                    prop_thresh=0, pad_length=args.text_max_length, verbase=0, ppl_num=args.ppl_num)

        self.train_dataloader_plain = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.vocab = train_dataset.vocab

        # val dataloader
        val_dataset = VQA2Dataset(split_dic_path=args.val_split_dic_path, vocab_path=args.vocab_path,
                                  split="val",
                                  prop_thresh=0, pad_length=args.text_max_length, verbase=0, ppl_num=args.ppl_num)

        self.val_sampler = None

        self.val_dataloader_plain = DataLoader(val_dataset, batch_size=args.batch_size, sampler=self.val_sampler,
                                         shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.val_dataloader = self.val_dataloader_plain # self.accelerator.prepare_data_loader(self.val_dataloader_plain)

        # test dataloader
        test_dataset = VQA2Dataset(split_dic_path=args.test_split_dic_path, vocab_path=args.vocab_path,
                                  split="test",
                                  prop_thresh=0, pad_length=args.text_max_length, verbase=0, ppl_num=args.ppl_num)

        self.test_sampler = None

        self.test_dataloader_plain = DataLoader(test_dataset, batch_size=args.batch_size, sampler=self.test_sampler,
                                          shuffle=False, num_workers=args.num_workers, pin_memory=True)
        self.test_dataloader = self.test_dataloader_plain # self.accelerator.prepare_data_loader(self.test_dataloader_plain)


    def __build_model(self, args):
        self.generator = Graph2seqGeneratorVhSamples.from_opts(args, vocab=self.vocab, device=self.device).to(self.device)
        self.discriminator = Discriminator(vocab=self.vocab, d_v=args.proposal_dim, d_word=args.word_dim, d_model=args.hidden_size).to(self.device)

    def __build_optimizer(self, args):
        params = []
        for key, value in dict(self.discriminator.named_parameters()).items():
            if value.requires_grad:
                if 'cnn' in key:
                    pass
                else:
                    params += [{'params': [value], 'lr': float(args.discriminator_learning_rate),
                                'weight_decay': float(args.weight_decay),
                                'betas': (float(args.optim_alpha), float(args.optim_beta))}]
        self.discriminator_optimizer = optim.Adagrad(params)
        self.discriminator, self.discriminator_optimizer, self.train_dataloader_discriminator = self.accelerator.prepare(self.discriminator, self.discriminator_optimizer, self.train_dataloader_plain)


        params = []
        for key, value in dict(self.generator.named_parameters()).items():
            if value.requires_grad:
                if 'cnn' in key:
                    pass
                else:
                    params += [{'params': [value], 'lr': float(args.generator_learning_rate),
                                'weight_decay': float(args.weight_decay),
                                'betas': (float(args.optim_alpha), float(args.optim_beta))}]
        self.generator_optimizer = optim.Adam(params)
        self.generator, self.generator_optimizer, self.train_dataloader_generator = self.accelerator.prepare(self.generator, self.generator_optimizer, self.train_dataloader_plain)

        # assert args.lr_scheduler == "ExponentialLR"
        # self.lr_scheduler_generator: optim.lr_scheduler.ExponentialLR = getattr(optim.lr_scheduler, args.lr_scheduler)(self.generator_optimizer, gamma=args.gamma)
        # self.lr_scheduler_discriminator: optim.lr_scheduler.ExponentialLR = getattr(optim.lr_scheduler, args.lr_scheduler)(self.discriminator_optimizer, gamma=args.gamma)
        # self.reset_lr(self.generator_optimizer, 5e-3)
        # del self.lr_scheduler_generator
        # self.lr_scheduler_generator: optim.lr_scheduler.ExponentialLR = getattr(optim.lr_scheduler, args.lr_scheduler)(self.generator_optimizer, gamma=args.gamma)

        # print(self.generator_optimizer.param_groups[-1]["lr"], "-------")
        # print(self.lr_scheduler_generator.get_last_lr()[-1])
        # exit(0)

    def __build_evaluator(self, args):
        self.metric = [str2metric["cider"](df="corpus"),
                       str2metric["bleu"](n_grams=[1, 2, 3, 4]),
                       str2metric["meteor"](),
                       str2metric["rouge"](),
                       str2metric["spice"](),
                       str2metric["accuracy"](metrics=["precision", "recall", "F1", "accuracy"])]

    def save_generator(self, epoch, save_dir):
        if self.accelerator.distributed_type == DistributedType.NO or (self.accelerator.distributed_type == DistributedType.MULTI_GPU and self.accelerator.is_local_main_process):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            args_filename = "args-generator-{}.pkl".format(epoch)
            model_name = "model-generator-{}.pth".format(epoch)
            with open(os.path.join(save_dir, args_filename), "wb") as f:
                pickle.dump(self.opt, f)
            unwrappered_generator = self.accelerator.unwrap_model(self.generator)
            self.accelerator.save(unwrappered_generator.state_dict(), os.path.join(save_dir, model_name))

    def save_discriminator(self, epoch, save_dir):
        if self.accelerator.distributed_type == DistributedType.NO or (self.accelerator.distributed_type == DistributedType.MULTI_GPU and self.accelerator.is_local_main_process):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            print("------", epoch)
            args_filename = "args-discriminator-{}.pkl".format(epoch)
            model_name = "model-discriminator-{}.pth".format(epoch)
            with open(os.path.join(save_dir, args_filename), "wb") as f:
                pickle.dump(self.opt, f)
            unwrappered_discriminator = self.accelerator.unwrap_model(self.discriminator)
            self.accelerator.save(unwrappered_discriminator.state_dict(), os.path.join(save_dir, model_name))


    def load_pretrain_generator(self, epoch, save_dir):
        model_name = "model-generator-{}.pth".format(epoch)
        checkpoint_path = os.path.join(save_dir, model_name)
        unwrappered_generator = self.accelerator.unwrap_model(self.generator)
        unwrappered_generator.load_state_dict(torch.load(checkpoint_path))


    def load_pretrain_discriminator(self, epoch, save_dir):
        model_name = "model-discriminator-{}.pth".format(epoch)
        checkpoint_path = os.path.join(save_dir, model_name)
        unwrappered_discriminator = self.accelerator.unwrap_model(self.discriminator)
        unwrappered_discriminator.load_state_dict(torch.load(checkpoint_path))
    
    
    def pretrain_generator(self, epoch, lr):
  
        self.generator.train()

        start = time.time()

        loss_generator_collect = []
        loss_lm_collect = []
        loss_adv_collect = []
        loss_vh_collect = []

        for step, data in enumerate(self.train_dataloader_generator):
            box_feats, box_info, visual_hint, question, answer, question_str, _ = data
            self.global_steps_generator += 1

            model_in = {"ppl_feats": box_feats, "ppl_info": box_info, "question": question, "answer": answer}
            
        
            loss_lm, vh_logits = self.generator(**model_in)
            
            loss_vh = self.loss_vh(pred=vh_logits, gt=visual_hint)

            loss_all = loss_lm + loss_vh*self.opt.vh_weight

            self.generator_optimizer.zero_grad()
            self.accelerator.backward(loss_all)
            self.generator_optimizer.step()

            loss_generator_collect.append(loss_all.item())
            loss_lm_collect.append(loss_lm.item())
            loss_vh_collect.append(loss_vh.item())
            if self.main_condition:
                self.writer.add_scalar("train/loss_pretrain_generator", scalar_value=loss_all.item(), global_step=self.global_steps_generator)
                self.writer.add_scalar("train/loss_pretrain_generator_lm", scalar_value=loss_lm.item(), global_step=self.global_steps_generator)
                self.writer.add_scalar("train/loss_pretrain_generator_vh", scalar_value=loss_vh.item(), global_step=self.global_steps_generator)

            if step % 100 == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "step {}/{} (epoch {}), Pre_training, generator_loss = {:.4f}, "
                    "lm_loss = {:.4f}, visual_hint_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
                    .format(step, len(self.train_dataloader_plain), epoch,
                    np.mean(loss_generator_collect), np.mean(loss_lm_collect),
                    np.mean(loss_vh_collect), float(lr), end - start))
                
                loss_generator_collect = []
                loss_lm_collect = []
                loss_vh_collect = []
                start = time.time()
        pass


    def pretrain_discriminator(self, epoch, lr):
        self.discriminator.train()
        self.generator.train()
        start = time.time()
        loss_discriminator_collect = []
        loss_generator_collect = []
        loss_lm_collect = []
        loss_adv_collect = []
        for step, data in enumerate(self.train_dataloader_discriminator):
            box_feats, box_info, visual_hint, question, answer, question_str, _ = self.to_cuda(data)
            self.global_steps_discriminator += 1

            model_in = {"ppl_feats": box_feats, "ppl_info": box_info, "question": None, "answer": answer, "sampling_procedure": True}

            logits, sampled_results, vh_logits, vh_pred_label = self.generator(**model_in)
            
            # train discriminator
            question_mask = (question != self.vocab.word2idx[self.vocab.SYM_PAD])
            ## real
            label = torch.ones((box_feats.shape[0])).to(self.device)
            model_in = {"visual_feats": box_feats, "visual_spatial_feats": box_info, "visual_hints": visual_hint,
                                    "question": question, "question_mask": question_mask, "answer": answer, "labels": label}
            loss_real = self.discriminator(**model_in)

            ## fake
            # sampled_results = logits.argmax(2)
            label = torch.zeros((box_feats.shape[0])).to(self.device)
            sampled_mask = sampled_results != 2

            model_in = {"visual_feats": box_feats, "visual_spatial_feats": box_info, "visual_hints": vh_pred_label.detach(),
                                    "question": sampled_results.detach(), "question_mask": sampled_mask.detach(), "answer": answer, "labels": label}
            loss_fake = self.discriminator(**model_in)
            
            loss_discriminator = (loss_real + loss_fake) / 2
            self.discriminator_optimizer.zero_grad()
            self.accelerator.backward(loss_discriminator)
            self.accelerator.clip_grad_norm_(self.discriminator.parameters(), 2)
            self.discriminator_optimizer.step()
            

            loss_discriminator_collect.append(loss_discriminator.item())
            if self.main_condition:
                self.writer.add_scalar("train/loss_pretrain_discriminator", scalar_value=loss_discriminator.item(), global_step=self.global_steps_discriminator)


            if step % 100 == 0 and step != 0:
                end = time.time()
                self.logger.info(
                    "[Adversarial Training][Discriminator] step {}/{} (epoch {}), discriminator_loss = {:.4f}, generator_loss = {:.4f}, "
                    "lm_loss = {:.4f}, adv_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
                    .format(step, len(self.train_dataloader_plain), epoch, np.mean(loss_discriminator_collect),
                    np.mean(loss_generator_collect), np.mean(loss_lm_collect),
                    np.mean(loss_adv_collect), float(lr), end - start))
                
                loss_discriminator_collect = []
                loss_generator_collect = []
                loss_lm_collect = []
                loss_adv_collect = []
                start = time.time()
        
        pass

    
    def train_generator_pg(self, epoch, lr):
        self.discriminator.train()
        self.generator.train()

        start = time.time()
        loss_discriminator_collect = []
        loss_generator_collect = []
        loss_lm_rl_collect = []
        loss_vh_rl_collect = []

        for step, data in enumerate(self.train_dataloader_generator):
            box_feats, box_info, visual_hint, question, answer, question_str, _ = data
            self.global_steps_generator += 1

            with torch.autograd.set_detect_anomaly(False):

                model_in = {"ppl_feats": box_feats.clone(), "ppl_info": box_info.clone(), "question": question.clone(), "answer": answer.clone()}

                loss_lm, vh_logits = self.generator(**model_in)
                loss_vh = self.loss_vh(pred=vh_logits, gt=visual_hint)
                loss_first = loss_lm + loss_vh*self.opt.vh_weight

                self.generator_optimizer.zero_grad()
                self.accelerator.backward((1 - self.opt.rl_ratial) * loss_first)

            
                # baseline
                with torch.no_grad():
                    model_in = {"ppl_feats": box_feats.clone(), "ppl_info": box_info.clone(), "question": None, "answer": answer.clone()}

                    logits_baseline, vh_pred_label_baseline = self.generator(**model_in)
                    question_baseline = logits_baseline.argmax(2)
                    sentence_baseline = self.vocab.convert_ids(question_baseline.detach().cpu().data)


                    model_in = {"visual_feats": box_feats, "visual_spatial_feats": box_info, "visual_hints": vh_pred_label_baseline.detach(),
                                            "question": question_baseline, "question_mask": question_baseline != 2, "answer": answer, "labels": None}

                    dis_logits = self.discriminator(**model_in)
                    reward_cons_baseline = torch.sigmoid(dis_logits)

                # explore
                
                model_in = {"ppl_feats": box_feats, "ppl_info": box_info, "question": None, "answer": answer, "sampling_procedure": True}
                logits, sampled_results, vh_logits, vh_pred_label = self.generator(**model_in)
                    # visual_hint_fake = torch.softmax(vh_logits, dim=1) >= 0.1
                
                sentence_explore = self.vocab.convert_ids(sampled_results.detach().cpu().data)

                with torch.no_grad():
                    model_in = {"visual_feats": box_feats, "visual_spatial_feats": box_info, "visual_hints": vh_pred_label,
                                            "question": sampled_results, "question_mask": sampled_results != 2, "answer": answer, "labels": None}

                    dis_logits = self.discriminator(**model_in)

                    reward_cons_explore = torch.sigmoid(dis_logits)
                
                reward_cons = reward_cons_explore - reward_cons_baseline
                
                bleu4_metric = []
                for i in range(vh_logits.shape[0]):
                    bleu4_baseline, _ = self.metric[1].calculate_scores(ground_truth=[question_str[i]], predict=[sentence_baseline[i]])
                    bleu4_explore, _ = self.metric[1].calculate_scores(ground_truth=[question_str[i]], predict=[sentence_explore[i]])
                    reward = bleu4_explore[3] - bleu4_baseline[3]
                    # print("reward bleu: ", reward)
                    # print("reward discriminator: ", reward_cons[i].item())
                    # print("gt: ", question_str[i])
                    # print("pred greed: ", sentence_baseline[i])
                    # print("pred explpre: ", sentence_explore[i])
                    # print("---------------------------")
                    bleu4_metric.append(reward)
                
                reward_bleu = torch.Tensor(bleu4_metric).to(vh_logits.device)
                
                reward_bleu_norm = reward_bleu
                reward_cons_norm = reward_cons

                reward = reward_bleu_norm + reward_cons_norm.detach() * self.opt.cons_weight

                # reward = reward_bleu_norm / reward_bleu_norm.abs().max().item() + 0.5 * reward_cons_norm / reward_cons_norm.abs().max().item()


                loss_lm_rl = pg_loss(prob=logits, gt=sampled_results.detach(), reward=reward.detach())
                loss_vh_rl = self.loss_vh(vh_logits, vh_pred_label.detach().long())
                loss_gen = self.opt.rl_ratial * ( loss_lm_rl + loss_vh_rl*self.opt.vh_weight*0.1) + (1 - self.opt.rl_ratial) * loss_first
                # print(loss_gen)

                # self.generator_optimizer.zero_grad()
                # self.accelerator.backward(loss_gen)
                loss2 = self.opt.rl_ratial * ( loss_lm_rl + loss_vh_rl*self.opt.vh_weight*0.1)
                self.accelerator.backward(loss2)
                self.accelerator.clip_grad_norm_(self.generator.parameters(), 2)
                self.generator_optimizer.step()
                
                loss_generator_collect.append(loss_gen.item())
                loss_lm_rl_collect.append(loss_lm_rl.item())
                loss_vh_rl_collect.append(loss_vh_rl.item())
                if self.main_condition:
                    self.writer.add_scalar("train/loss_generator", scalar_value=loss_gen.item(), global_step=self.global_steps_generator)
                    self.writer.add_scalar("train/adversarial_loss_generator_lm", scalar_value=loss_lm_rl.item(), global_step=self.global_steps_generator)
                    self.writer.add_scalar("train/adversarial_loss_generator_vh", scalar_value=loss_vh_rl.item(), global_step=self.global_steps_generator)



                if step % 100 == 0 and step != 0:
                    end = time.time()
                    self.logger.info(
                        "[Adversarial Training][Generator] step {}/{} (epoch {}), generator_loss = {:.4f}, "
                        "lm_loss = {:.4f}, visual_hint_loss = {:.4f}, lr = {:.5f}, time/batch = {:.4f}"
                        .format(step, len(self.train_dataloader_plain), epoch,
                        np.mean(loss_generator_collect), np.mean(loss_lm_rl_collect),
                        np.mean(loss_vh_rl_collect), float(lr), end - start))
                    
                    loss_discriminator_collect = []
                    loss_generator_collect = []
                    loss_lm_rl_collect = []
                    loss_vh_rl_collect = []
                    start = time.time()
            pass

    @torch.no_grad()
    def evaluate_generator(self, epoch=None, split="val", save=False):
        if self.accelerator.distributed_type == DistributedType.NO or (self.accelerator.distributed_type == DistributedType.MULTI_GPU and self.accelerator.is_local_main_process):
            self.generator.eval()
            start = time.time()

            assert split in ["val", "test"]

            dataloader = self.val_dataloader if split == "val" else self.test_dataloader

            pred_collect = []
            gt_collect = []

            vh_pred_student_collect = []
            vh_gt_collect = []

            cnt = 0

            with torch.no_grad():
                for step, data in enumerate(dataloader):

                    box_feats, box_info, visual_hint, question, answer, question_gt, question_idx = self.to_cuda(data)
                    
                    model_in = {"ppl_feats": box_feats, "ppl_info": box_info, "question": None, "answer": answer}
                    
                    prob, visual_hint_pred = self.generator(**model_in)

                    prob_ids = prob.argmax(dim=-1)
                    sentence_pred = self.vocab.convert_ids(prob_ids.detach().cpu().data)
                    sentence_gt = question_gt
                    pred_collect.extend(sentence_pred)
                    gt_collect.extend(sentence_gt)

                    # visual hint
                    vh_pred_student_collect.append(visual_hint_pred.detach().cpu())
                    vh_gt_collect.append(visual_hint.detach().cpu())


                
                end = time.time()
                if self.accelerator.is_local_main_process:
                    if save:
                        if not os.path.exists(self.opt.save_dir):
                            os.makedirs(self.opt.save_dir, exist_ok=True)
                        cpt_filename = "results-generator-{}-{}.pkl".format(split, epoch)

                        import pickle
                        dump_file = []
                        ans_str = self.vocab.convert_ids(answer.detach().cpu().data)
                        for i in range(box_feats.shape[0]):
                            dump_file.append(
                                {
                                    "question_id": question_idx[i],
                                    "visual_hint_pred": visual_hint_pred.detach().cpu().numpy()[i],
                                    "visual_hint_gt": visual_hint.detach().cpu().numpy()[i],
                                    "pred_question": sentence_pred[i],
                                    "gt_question": sentence_gt[i],
                                    "answer": ans_str[i]
                                }
                            )
                        for i in range(5):
                            self.logger.info("Samples: i ======================")
                            for key, val in dump_file[i].items():
                                self.logger.info("{} : {}".format(key, val))
                            self.logger.info("----------------------")
                        with open(os.path.join(self.opt.save_dir, cpt_filename), "wb") as f:
                            pickle.dump(dump_file, f)

                    self.logger.info("*********** Evaluation, split: {} ***********".format(split))
                    self.logger.info("Time cost: {:.4f}".format(end - start))

                    
                    vh_pred_student_collect_concat = torch.cat(vh_pred_student_collect, dim=0).view(-1).int()
                    vh_gt_collect_concat = torch.cat(vh_gt_collect, dim=0).view(-1).int()
                    scores = self.metric[5].calculate_scores(ground_truth=vh_gt_collect_concat,
                                                            predict=vh_pred_student_collect_concat)
                    self.logger.info("Visual Hint Prediction performance")
                    self.logger.info("Metric Precision: 0 - {}, 1 - {}".format(scores[0][0], scores[0][1]))
                    self.logger.info("Metric Recall: 0 - {}, 1 - {}".format(scores[1][0], scores[1][1]))
                    self.logger.info("Metric F1: 0 - {}, 1 - {}".format(scores[2][0], scores[2][1]))
                    self.logger.info("Metric Accuracy: {}".format(scores[3]))

                    score, scores = self.metric[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
                    print(scores)
                    self.logger.info("Metric {}: {:.4f}".format(str(self.metric[0]), score))
                    self.writer.add_scalar(split + "/CIDEr", score * 100, global_step=epoch)

                    score, _ = self.metric[1].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
                    self.logger.info("Metric {}: @1 - {:.4f}, @2 - {:.4f}, @3 - {:.4f}, @4 - {:.4f}".format(
                        str(self.metric[1]), score[0], score[1], score[2], score[3]))
                    self.writer.add_scalar(split + "/BLEU@1", score[0] * 100, global_step=epoch)
                    self.writer.add_scalar(split + "/BLEU@2", score[1] * 100, global_step=epoch)
                    self.writer.add_scalar(split + "/BLEU@3", score[2] * 100, global_step=epoch)
                    self.writer.add_scalar(split + "/BLEU@4", score[3] * 100, global_step=epoch)

                    bleu4 = score[3]

                    score, _ = self.metric[2].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
                    self.logger.info("Metric {}: {:.4f}".format(str(self.metric[2]), score))
                    self.writer.add_scalar(split + "/METEOR", score * 100, global_step=epoch)

                    score, _ = self.metric[3].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
                    self.logger.info("Metric {}: {:.4f}".format(str(self.metric[3]), score))
                    self.writer.add_scalar(split + "/ROUGE", score * 100, global_step=epoch)

                    score, _ = self.metric[4].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
                    self.logger.info("Metric {}: {:.4f}".format(str(self.metric[4]), score))
                    self.writer.add_scalar(split + "/SPICE", score * 100, global_step=epoch)
            return bleu4
        return

    @torch.no_grad()
    def evaluate_discriminator(self, epoch=None, split="val", save_dir="results"):
        if self.accelerator.distributed_type == DistributedType.NO or (self.accelerator.distributed_type == DistributedType.MULTI_GPU and self.accelerator.is_local_main_process):
            self.discriminator.eval()
            self.generator.eval()
            start = time.time()

            assert split in ["val", "test"]

            dataloader = self.val_dataloader if split == "val" else self.test_dataloader

            pred_collect = []
            gt_collect = []

            cnt = 0

            correct_fake = 0
            all_cnt_fake = 0
            correct_gt = 0
            all_cnt_gt = 0

            with torch.no_grad():
                for step, data in enumerate(dataloader):
                    if step % 100 == 0 and step != 0:
                        print("Step: {}/{}".format(step, len(dataloader)))
                    cnt += 1

                    box_feats, box_info, visual_hint, question, answer, question_gt, question_idx = self.to_cuda(data)
                    model_in = {"ppl_feats": box_feats, "ppl_info": box_info, "question": None, "answer": answer}
                    
                    logits, sampled_results, _, vh_pred_label = self.generator.module.sample(**model_in)

                    # visual hint


                    # visual_hint, label = self.make_fake_visual_hints(visual_hints=visual_hint)
                    question_mask = (question != self.vocab.word2idx[self.vocab.SYM_PAD])


                    sampled_mask = sampled_results != 2


                    model_in = {"visual_feats": box_feats, "visual_spatial_feats": box_info, "visual_hints": vh_pred_label,
                            "question": sampled_results, "question_mask": sampled_mask, "answer": answer, "labels": None}
                    prob = self.discriminator(**model_in)
                    
                    pred = torch.sigmoid(prob) >= 0.5

                    correct_fake += (pred == False).sum()
                    all_cnt_fake += pred.shape[0]


                    

                    model_in = {"visual_feats": box_feats, "visual_spatial_feats": box_info, "visual_hints": visual_hint,
                            "question": question, "question_mask": question_mask, "answer": answer, "labels": None}
                    prob = self.discriminator(**model_in)
                    
                    pred = torch.sigmoid(prob) >= 0.5

                    correct_gt += (pred).sum()
                    all_cnt_gt += pred.shape[0]

                
                end = time.time()

                self.logger.info("*********** Evaluation, split: {} ***********".format(split))
                self.logger.info("Time cost: {:.4f}".format(end - start))

                self.logger.info("Discriminator Accuracy Generator: {}".format(correct_fake / all_cnt_fake))
                self.logger.info("Discriminator Accuracy GT: {}".format(correct_gt / all_cnt_gt))

                self.writer.add_scalar(split + "/discriminator_accuracy Generator", (correct_fake / all_cnt_fake) * 100, global_step=epoch)
                self.writer.add_scalar(split + "/discriminator_accuracy GT", (correct_gt / all_cnt_gt) * 100, global_step=epoch)
            return (correct_fake + correct_gt) / (all_cnt_fake + all_cnt_gt)
    def to_cuda(self, data):
        ret = []
        for x in data:
            if isinstance(x, torch.Tensor):
                ret.append(x.to(self.device))
            else:
                ret.append(x)
        return ret

    def reset_lr(self, opt, lr):
        for params_group in opt.param_groups:
            params_group["lr"] = lr

    @property
    def main_condition(self):
        return self.accelerator.distributed_type == DistributedType.NO or (self.accelerator.distributed_type == DistributedType.MULTI_GPU and self.accelerator.is_local_main_process)

    def train_generator(self):
        if self.opt.generator_need_pretrain:
            self.global_steps_generator = 0
            self.reset_lr(self.generator_optimizer, lr=self.opt.pretrain_generator_learning_rate)
            best_bleu = -1
            for epoch in range(self.opt.generator_pretrain_epoch):
                self.pretrain_generator(epoch=epoch, lr=self.opt.pretrain_generator_learning_rate)
                if self.main_condition:
                    bleu4 = self.evaluate_generator(split="val", epoch=epoch)
                    if best_bleu < bleu4:
                        best_bleu = bleu4
                        
                        best_epoch = epoch
                        self.logger.info("BLEU metric enhanced !!!!")
                        self.save_generator(save_dir=self.opt.generator_pretrain_checkpoint_path, epoch=epoch)
                        with open(self.opt.name + "best_epoch.pkl", "wb") as f:
                            pickle.dump(best_epoch, f)
            self.accelerator.wait_for_everyone()
            with open(self.opt.name + "best_epoch.pkl", "rb") as f:
                best_epoch = pickle.load(f)
            
            self.load_pretrain_generator(epoch=best_epoch, save_dir=self.opt.generator_pretrain_checkpoint_path)
            self.evaluate_generator(split="test", epoch=-1)
        else:
            self.load_pretrain_generator(epoch=self.opt.start_generator_epoch, save_dir=self.opt.generator_pretrain_checkpoint_path)
            self.evaluate_generator(split="test", epoch=-1)

    def train_discriminator(self):
        if self.opt.discriminator_need_pretrain:
            self.global_steps_discriminator = 0
            self.reset_lr(self.discriminator_optimizer, lr=self.opt.pretrain_discriminator_learning_rate)
            best_acc = -1
            best_epoch = -1
            for epoch in range(self.opt.discriminator_pretrain_epoch):
                self.pretrain_discriminator(epoch=epoch, lr=self.opt.pretrain_discriminator_learning_rate)
                
                if self.main_condition:
                    acc = self.evaluate_discriminator(split="val", epoch=epoch)
                    if best_acc < acc:
                        best_acc = acc
                        best_epoch = epoch
                        self.save_discriminator(save_dir=self.opt.discriminator_pretrain_checkpoint_path, epoch=epoch)

            self.accelerator.wait_for_everyone()
            self.load_pretrain_discriminator(epoch=0, save_dir=self.opt.discriminator_pretrain_checkpoint_path)
            acc = self.evaluate_discriminator(split="test", epoch=-1)
        else:
            self.load_pretrain_discriminator(epoch=self.opt.start_discriminator_epoch, save_dir=self.opt.discriminator_pretrain_checkpoint_path)
            acc = self.evaluate_discriminator(split="test", epoch=-1)

    def train(self):
        self.train_generator()
        self.train_discriminator()
        self.reset_lr(self.generator_optimizer, lr=self.opt.generator_learning_rate)
        self.reset_lr(self.discriminator_optimizer, lr=self.opt.discriminator_learning_rate)

        # adversirial training
        self.global_steps_generator = 0
        self.global_steps_discriminator = 0
        best_epoch = -1
        best_bleu4 = -1
        for epoch in range(self.opt.epoch_all):
            if epoch > self.opt.lr_decay_epoch_start and epoch % self.opt.lr_decay_epoch_num == 0:
                set_lr(self.generator_optimizer, self.opt.gamma)
                set_lr(self.discriminator_optimizer, self.opt.gamma)
                self.opt.generator_learning_rate = self.opt.generator_learning_rate * self.opt.gamma
                self.opt.discriminator_learning_rate = self.opt.discriminator_learning_rate * self.opt.gamma
            self.train_generator_pg(epoch, lr=self.opt.generator_learning_rate)
            if self.main_condition:
                score = self.evaluate_generator(split="val", epoch=epoch, save=False)
                self.save_generator(save_dir=self.opt.generator_checkpoint_path, epoch=epoch)
                self.save_discriminator(save_dir=self.opt.discriminator_checkpoint_path, epoch=epoch)
                if score > best_bleu4:
                    best_bleu4 = score
                    best_epoch = epoch
                    self.logger.info("Updated in val !!!!")
                    self.evaluate_generator(split="test", epoch=best_epoch, save=True)
                    self.evaluate_discriminator(split="test", epoch=best_epoch)
                
            if epoch % 2 == 0 and epoch != 0:
                self.pretrain_discriminator(epoch, lr=self.opt.discriminator_learning_rate)


