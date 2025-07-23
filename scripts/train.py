#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import time
from tqdm import tqdm

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'DomainBed'))

from dangr import DANGR
from dangr.utils import set_random_seed, save_checkpoint, create_dir

try:
    from DomainBed.domainbed import datasets
    from DomainBed.domainbed.lib import misc
    from DomainBed.domainbed import hparams_registry
except ImportError:
    print("Error: DomainBed not found. Please make sure DomainBed is in the correct path.")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description='Train DANGR model')
    parser.add_argument('--data_dir', type=str, default='./DomainBed/data')
    parser.add_argument('--dataset', type=str, default='PACS', 
                       choices=['PACS', 'VLCS', 'OfficeHome', 'TerraIncognita', 'DomainNet'])
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--hparams', type=str, default='{}',
                       help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                       help='Seed for random hyperparameter sampling')
    parser.add_argument('--trial_seed', type=int, default=0,
                       help='Trial number (used for seeding random_hparams).')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                       help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    return parser.parse_args()

def load_config(args):
    """설정 파일 로드 및 args와 병합"""
    config = {}
    
    # config 파일이 지정되면 로드
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # args가 config보다 우선
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    
    # hparams 문자열 파싱
    if isinstance(config.get('hparams'), str):
        config['hparams'] = json.loads(config['hparams'])
    
    return config

def train(args):
    """메인 학습 함수"""
    # 설정 로드
    config = load_config(args)
    
    # 시드 설정
    set_random_seed(config['seed'])
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(
        config['output_dir'],
        f"{config['dataset']}_test{config['test_envs'][0]}_seed{config['seed']}"
    )
    create_dir(output_dir)
    
    # 로그 파일 설정
    sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(output_dir, 'err.txt'))
    
    print("Environment:")
    print("\tPython:", sys.version.split(" ")[0])
    print("\tPyTorch:", torch.__version__)
    print("\tCUDA:", torch.cuda.is_available())
    print("\tCUDNN:", torch.backends.cudnn.enabled)
    print("\tNumPy:", np.__version__)
    print("")
    print("Args:", config)
    print("")
    
    # 하이퍼파라미터 설정
    if config['dataset'] in vars(hparams_registry):
        hparams = hparams_registry.default_hparams('DANGR', config['dataset'])
    else:
        hparams = hparams_registry.default_hparams('ERM', config['dataset'])
    
    hparams.update(config.get('hparams', {}))
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print(f'\t{k}: {v}')
    print("")
    
    # 데이터셋 로드
    if config['dataset'] in vars(datasets):
        dataset_class = vars(datasets)[config['dataset']]
    else:
        raise NotImplementedError(f"Dataset {config['dataset']} not implemented")
    
    dataset = dataset_class(config['data_dir'], config['test_envs'], hparams)
    
    # 데이터로더 생성
    train_loaders = []
    val_loaders = []
    test_loaders = []
    
    for i, env in enumerate(dataset):
        if i in config['test_envs']:
            loader = DataLoader(
                env,
                batch_size=hparams['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            test_loaders.append(loader)
        else:
            # 학습/검증 분할 (80/20)
            n = len(env)
            n_train = int(0.8 * n)
            n_val = n - n_train
            train_env, val_env = torch.utils.data.random_split(env, [n_train, n_val])
            
            train_loader = DataLoader(
                train_env,
                batch_size=hparams['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
            val_loader = DataLoader(
                val_env,
                batch_size=hparams['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
    
    # DANGR 모델 초기화
    algorithm = DANGR(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(config['test_envs']),
        hparams
    )
    algorithm.to(algorithm.device)
    
    # 학습 설정
    n_steps = config.get('steps') or dataset.N_STEPS
    checkpoint_freq = config.get('checkpoint_freq') or dataset.CHECKPOINT_FREQ
    
    # 메트릭 추적
    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = defaultdict(lambda: [])
    steps_per_epoch = min(len(loader) for loader in train_loaders)
    n_epochs = n_steps // steps_per_epoch + 1
    
    # 학습 루프
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # 에포크별 메트릭 초기화
        epoch_stats = defaultdict(list)
        
        # 학습
        algorithm.train()
        for batch_idx in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{n_epochs}"):
            step = epoch * steps_per_epoch + batch_idx
            if step >= n_steps:
                break
                
            try:
                minibatches = next(train_minibatches_iterator)
            except StopIteration:
                train_minibatches_iterator = zip(*train_loaders)
                minibatches = next(train_minibatches_iterator)
            
            # 데이터를 디바이스로 이동
            minibatches = [(x.to(algorithm.device), y.to(algorithm.device)) 
                          for x, y in minibatches]
            
            # 모델 업데이트
            step_vals = algorithm.update(minibatches)
            
            # 메트릭 기록
            for key, val in step_vals.items():
                epoch_stats[key].append(val)
            
            # 체크포인트
            if (step + 1) % checkpoint_freq == 0:
                print(f"\nStep {step + 1}/{n_steps}")
                
                # 검증 정확도 계산
                algorithm.eval()
                val_accs = []
                with torch.no_grad():
                    for val_loader in val_loaders:
                        correct = 0
                        total = 0
                        for x, y in val_loader:
                            x, y = x.to(algorithm.device), y.to(algorithm.device)
                            logits = algorithm.predict(x)
                            predictions = logits.argmax(dim=1)
                            correct += (predictions == y).sum().item()
                            total += y.size(0)
                        val_accs.append(correct / total)
                
                val_acc = np.mean(val_accs)
                checkpoint_vals['step'].append(step)
                checkpoint_vals['val_acc'].append(val_acc)
                
                print(f"Validation accuracy: {val_acc:.4f}")
                
                # 평균 손실 출력
                for key, values in epoch_stats.items():
                    if values:
                        print(f"  {key}: {np.mean(values):.4f}")
                
                # 최고 모델 저장
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if not config.get('skip_model_save'):
                        save_path = os.path.join(output_dir, 'best_model.pth')
                        save_checkpoint({
                            'step': step,
                            'epoch': epoch,
                            'model_state_dict': algorithm.get_state_dict(),
                            'best_val_acc': best_val_acc,
                            'hparams': hparams
                        }, save_path)
                        print(f"Saved best model with val_acc: {best_val_acc:.4f}")
                
                algorithm.train()
        
        # 학습률 스케줄러 스텝
        algorithm.step_schedulers()
    
    # 최종 테스트
    algorithm.eval()
    print("\nFinal evaluation:")
    
    # 테스트 정확도
    test_accs = []
    with torch.no_grad():
        for i, test_loader in enumerate(test_loaders):
            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(algorithm.device), y.to(algorithm.device)
                logits = algorithm.predict(x)
                predictions = logits.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
            
            test_acc = correct / total
            test_accs.append(test_acc)
            print(f"Test env {config['test_envs'][i]} accuracy: {test_acc:.4f}")
    
    print(f"\nAverage test accuracy: {np.mean(test_accs):.4f}")
    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
    
    # 결과 저장
    results = {
        'config': config,
        'hparams': hparams,
        'test_accs': test_accs,
        'best_val_acc': best_val_acc,
        'checkpoint_vals': dict(checkpoint_vals)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDone!")

if __name__ == '__main__':
    args = get_args()
    train(args)
