import os
import sys
import wandb
import torch
import datetime

import speechbrain as sb
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml

from transformers import get_constant_schedule_with_warmup

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class ListenChatRemix(sb.Brain):
    
    def compute_text_embedding(self, texts):
        tokens = self.hparams.tokenizer(texts, padding=True, return_tensors='pt')['input_ids'].to(self.device)
            
        words_embed = self.hparams.lora_llm(
            tokens, output_hidden_states=True
        ).hidden_states[-1] # last layer
        
        return words_embed[:, -1, :] # last or EOS token
            
    def compute_forward(self, mix, prompt, stage):
        
        # Encoding text
        text_embed = self.compute_text_embedding(prompt)
        if stage != sb.Stage.TRAIN:
            text_embed = text_embed.float()
        
        # Encoding speech
        mix_h = self.hparams.Encoder(mix)
        
        # Editing
        est_mask = self.hparams.MaskNet(mix_h, text_embed) #  (1, B, F, T)
        if est_mask.shape[0] == 1: # one mask by default
            est_mask = est_mask.squeeze(0)
            est_tar_h = mix_h * est_mask # (B, F, T)
        else: # ablation study: estimate multiple targets and sum them together
            est_tar_h = sum([mix_h * est_mask[i] for i in range(est_mask.shape[0])])

        # Decoding
        est_tar = self.hparams.Decoder(est_tar_h)
        
        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_ext = est_tar.size(1)
        
        if T_origin > T_ext:
            est_tar = F.pad(est_tar, (0, T_origin - T_ext))
        else:
            est_tar = est_tar[:, :T_origin]
            
        return est_tar
    

    def compute_objectives(self, est_tar, tar, mix, task):
        
        B = est_tar.shape[0]

        if self.hparams.loss_fn == 'SNR':
            snr = self.hparams.SNR(est_tar, tar, zero_mean=True)
            base_snr = self.hparams.SNR(mix, tar, zero_mean=True)
        elif self.hparams.loss_fn == 'SISNR':
            snr = self.hparams.SISNR(est_tar, tar)
            base_snr = self.hparams.SISNR(mix, tar)
        
        # Filter out NaN
        nan_mask = torch.isnan(snr)
        snr = torch.masked_select(snr, ~nan_mask)
        base_snr = torch.masked_select(base_snr, ~nan_mask)    
        task = [task[i] for i, ifnan in enumerate(nan_mask) if not ifnan]
        impr_snr = snr - base_snr 

        for _task, _snr, _impr_snr in zip(task, snr, impr_snr):
            self.loss_stat_tasks[_task]['snr'] += float(_snr)
            self.loss_stat_tasks[_task]['snri'] += float(_impr_snr)
            self.count_tasks[_task] += 1
        
        loss = -impr_snr.mean()
        nan_ratio = B - snr.shape[0]
        if nan_ratio:
            print(f'NaN occurs in {str(nan_ratio)}/{str(B)} losses in the batch!')
                
        loss_dict = {
            'loss': loss,
            'snr': snr.mean(),
            'snri': impr_snr.mean(),
            'nan_ratio': nan_ratio
        }
        
        # Update loss stat
        if not torch.isnan(loss):
            self.count += B
            for key in self.loss_stat:
                with torch.no_grad():
                    self.loss_stat[key] += B * loss_dict[key]
                    
        return loss, loss_dict

    
    def on_fit_start(self):
        super().on_fit_start()
        self.global_fit_step = 0
        
        
    def init_optimizers(self):
            
        if self.opt_class is not None:
            lora_prefix = (
                'lora_llm',
                'module.lora_llm',
            )
            optim_params = [{
                'name': 'extractor',
                'params': [param for name, param in self.modules.named_parameters() if not name.startswith(lora_prefix)],
                'lr': self.hparams.lr,
            }, {
                'name': 'lora_llm',
                'params': [param for name, param in self.modules.named_parameters() if name.startswith(lora_prefix)],
                'lr': self.hparams.lr_lora,
                'weight_decay': self.hparams.wd_lora
            }]
        
            print(f'Initialized Optimizer {str(self.opt_class)}:')
            print(f'Extractor lr: {str(self.hparams.lr)}')
            print(f'LLM LoRA lr: {str(self.hparams.lr_lora)} wd: {str(self.hparams.wd_lora)}')
            
            self.optimizer = self.opt_class(optim_params)
            self.lr_scheduler = self.hparams.lr_scheduler(self.optimizer)
            
            if self.hparams.n_warmup_step != 0: # Linear warmup
                self.warmup_scheduler = get_constant_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=self.hparams.n_warmup_step
                )

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)
                self.checkpointer.add_recoverable("lr_scheduler", self.lr_scheduler)

            
    def fit_batch(self, batch):
        mix = batch['mix'].to(self.device)
        tar = batch['tar'].to(self.device)
        prompt = batch['prompt']
        task = batch['task']
        
        if self.auto_mix_prec:
            with autocast(dtype=self.hparams.mix_dtype): # torch.bfloat16
                est_tar = self.compute_forward(mix, prompt, sb.Stage.TRAIN)
                loss, loss_dict = self.compute_objectives(est_tar, tar, mix, task)
        else:
            est_tar = self.compute_forward(mix, prompt, sb.Stage.TRAIN)
            loss, loss_dict = self.compute_objectives(est_tar, tar, mix, task)

        if (
            loss < self.hparams.loss_upper_lim and loss.nelement() > 0
        ):
            loss.backward()
            if self.hparams.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.hparams.clip_grad_norm
                )
            self.optimizer.step()
        else:
            self.nonfinite_count += 1
            print(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0).float().to(self.device)

        self.optimizer.zero_grad()
        if self.global_fit_step < self.hparams.n_warmup_step:
            self.warmup_scheduler.step()

        self.global_fit_step += 1

        return loss.detach().cpu()
    
    
    def evaluate_batch(self, batch, stage):
        mix = batch['mix'].to(self.device)
        tar = batch['tar'].to(self.device)
        prompt = batch['prompt']
        task = batch['task']

        with torch.no_grad():
            est_tar = self.compute_forward(mix, prompt, stage)
            loss, loss_dict = self.compute_objectives(est_tar, tar, mix, task)
 
        return loss.mean().detach()

    
    def on_stage_start(self, stage, epoch=None):
        self.loss_stat_tasks = {task: {'snr': 0, 'snri':0} for task in self.tasks}
        self.count_tasks = {task: 0 for task in self.tasks}
        
        self.loss_stat = {
            'nan_ratio': 0,
            'loss': 0,
            'snr': 0,
            'snri': 0,
        }
        self.count = 0
        
    
    def on_stage_end(self, stage, stage_loss, epoch):
        
        for key in self.loss_stat:
            if self.count > 0:
                self.loss_stat[key] /= self.count
                
        for task in self.tasks:
            if self.count_tasks[task] > 0:
                for key in self.loss_stat_tasks[task]: 
                    self.loss_stat_tasks[task][key] /= self.count_tasks[task]
                
        if stage == sb.Stage.TRAIN:
            stage_stats = {'train_'+key: round(float(value), 4) for key, value in self.loss_stat.items()}
            stage_stats['epoch'] = epoch
            for task in self.tasks:
                for key in self.loss_stat_tasks[task]:
                    if self.count_tasks[task] > 0:
                        stage_stats['train_'+key+'_'+task] = round(float(self.loss_stat_tasks[task][key]), 4)
            
    
        elif stage == sb.Stage.VALID:
            # Reduce LR on plateau
            if self.global_fit_step >= self.hparams.n_warmup_step:
                self.lr_scheduler.step(self.loss_stat['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            stage_stats = {'valid_'+key: round(float(value), 4) for key, value in self.loss_stat.items()}
            stage_stats['epoch'] = epoch
            stage_stats['lr'] = self.optimizer.param_groups[0]['lr']
            for task in self.tasks:
                for key in self.loss_stat_tasks[task]:
                    if self.count_tasks[task] > 0:
                        stage_stats['valid_'+key+'_'+task] = round(float(self.loss_stat_tasks[task][key]), 4)
                
            self.checkpointer.save_and_keep_only(
                meta=self.loss_stat, max_keys=['snri'],
            )

        elif stage == sb.Stage.TEST:
            stage_stats = {'test_'+key: round(float(value), 4) for key, value in self.loss_stat.items()}
            for task in self.tasks:
                for key in self.loss_stat_tasks[task]:
                    if self.count_tasks[task] > 0:
                        stage_stats['test_'+key+'_'+task] = round(float(self.loss_stat_tasks[task][key]), 4)

        if self.hparams.use_wandb:
            self.hparams.logger.run.log(
                data=stage_stats,
            )
            
        print(f'Epoch {epoch}: ', stage, stage_stats)


if __name__ == '__main__':

    argv = sys.argv[1:]
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d+%H-%M-%S')
    print(f'Experiment Time Stamp: {time_stamp}')
    argv += ['--time_stamp', time_stamp]

    hparam_file, run_opts, overrides = sb.parse_arguments(argv)

    with open(hparam_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
        
    run_opts['auto_mix_prec'] = hparams['mix_prec']

    # Initialize wandb

    if hparams['use_wandb']:
        hparams['logger'] = hparams['wandb_logger']()
        
    sb.utils.distributed.ddp_init_group(run_opts)

    # Config LLM and tokenizer

    hparams['tokenizer'].pad_token = '[PAD]'

    if hparams['llm_mix_prec']: # Cast LLM to bf16
        hparams['llm'] = hparams['llm'].to(hparams['mix_dtype'])

    for p in hparams['llm'].parameters():
        p.requires_grad = False
    print('Freezed LLM.')

    print('LLM LoRA at ', hparams['lora_modules'])

    # Load pretrained weights

    hparams['pretrainer'].collect_files()
    hparams['pretrainer'].load_collected()

    from utils.lora_ckpt import load_lora
    load_lora(
        lora_model=hparams['lora_llm'], 
        path=os.path.join(hparams['pretrain_folder'], 'lora_llm.ckpt'),
        end_of_epoch=None,
        device=hparams['lora_llm'].device
    )
    hparams['lora_llm'] = hparams['lora_llm'].to(run_opts['device'])
    print('Loaded pretrained weights.')

    # Main

    lcr = ListenChatRemix(
        modules=hparams['modules'],
        opt_class=hparams['optimizer'],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams['checkpointer'],
    )

    lcr.tasks = ['TAE', 'TAR']

    lcr.fit(
        epoch_counter=lcr.hparams.epoch_counter,
        train_set=hparams['train_set'],
        valid_set=hparams['valid_set'],
        train_loader_kwargs=hparams['train_loader_opts'],
        valid_loader_kwargs=hparams['valid_loader_opts'],
    )

    lcr.evaluate(
        test_set=hparams['test_set'],
        test_loader_kwargs=hparams['test_loader_opts'],
        max_key='snri'
    )

    wandb.finish()
