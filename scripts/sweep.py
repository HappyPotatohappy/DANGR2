#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
import itertools
import numpy as np
from collections import defaultdict
import time

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_args():
    parser = argparse.ArgumentParser(description='Run DANGR experiments sweep')
    parser.add_argument('--data_dir', type=str, default='./DomainBed/data')
    parser.add_argument('--dataset', type=str, default='PACS',
                       choices=['PACS', 'VLCS', 'OfficeHome', 'TerraIncognita', 'DomainNet'])
    parser.add_argument('--output_dir', type=str, default='./experiments/sweep')
    parser.add_argument('--command', type=str, default='train')
    parser.add_argument('--n_trials', type=int, default=3,
                       help='Number of random trials per test environment')
    parser.add_argument('--n_hparams', type=int, default=20,
                       help='Number of hyperparameter settings to sample')
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip experiments that already have results')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (only print commands)')
    parser.add_argument('--hparams', type=str, default='{}',
                       help='JSON-serialized hparams dict to use as base')
    
    return parser.parse_args()

def get_test_envs(dataset):
    """각 데이터셋의 테스트 환경 반환"""
    test_envs = {
        'PACS': [0, 1, 2, 3],  # Art, Cartoon, Photo, Sketch
        'VLCS': [0, 1, 2, 3],  # Caltech, LabelMe, SUN, VOC
        'OfficeHome': [0, 1, 2, 3],  # Art, Clipart, Product, Real
        'TerraIncognita': [0, 1, 2, 3],  # 4 locations
        'DomainNet': [0, 1, 2, 3, 4, 5]  # 6 domains
    }
    return test_envs.get(dataset, [0])

def sample_hparams(base_hparams, n_hparams, seed):
    """하이퍼파라미터 샘플링"""
    np.random.seed(seed)
    
    hparam_ranges = {
        'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'lr_g': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        'batch_size': [16, 32, 64],
        'weight_decay': [0, 1e-6, 1e-5, 1e-4],
        'lambda_domain': [0.1, 0.3, 0.5, 0.7, 1.0],
        'lambda_class': [0.5, 0.7, 1.0, 1.5, 2.0],
        'lambda_noise': [0.01, 0.05, 0.1, 0.2, 0.3],
        'dropout': [0.0, 0.1, 0.5],
        'data_augmentation': [True, False]
    }
    
    sampled_hparams = []
    for _ in range(n_hparams):
        hparams = base_hparams.copy()
        for key, values in hparam_ranges.items():
            hparams[key] = np.random.choice(values)
        sampled_hparams.append(hparams)
    
    return sampled_hparams

def run_experiment(command_args):
    """단일 실험 실행"""
    command, args, debug = command_args
    
    if debug:
        print(" ".join([command] + args))
        return {'status': 'debug', 'command': " ".join([command] + args)}
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, command] + args,
            capture_output=True,
            text=True,
            check=True
        )
        elapsed_time = time.time() - start_time
        
        return {
            'status': 'success',
            'elapsed_time': elapsed_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            'status': 'failed',
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }

def create_job_list(args):
    """실행할 작업 목록 생성"""
    jobs = []
    
    test_envs = get_test_envs(args.dataset)
    base_hparams = json.loads(args.hparams)
    
    # 하이퍼파라미터 샘플링
    hparam_list = sample_hparams(base_hparams, args.n_hparams, seed=0)
    
    for test_env in test_envs:
        for trial_seed in range(args.n_trials):
            for hparam_seed, hparams in enumerate(hparam_list):
                output_dir = os.path.join(
                    args.output_dir,
                    f"{args.dataset}_env{test_env}_trial{trial_seed}_hparams{hparam_seed}"
                )
                
                # 이미 결과가 있는지 확인
                if args.skip_existing and os.path.exists(os.path.join(output_dir, 'results.json')):
                    print(f"Skipping existing: {output_dir}")
                    continue
                
                command_args = [
                    '--dataset', args.dataset,
                    '--data_dir', args.data_dir,
                    '--test_envs', str(test_env),
                    '--output_dir', output_dir,
                    '--hparams', json.dumps(hparams),
                    '--hparams_seed', str(hparam_seed),
                    '--trial_seed', str(trial_seed),
                    '--seed', str(trial_seed)
                ]
                
                if args.command == 'train':
                    script_path = os.path.join(os.path.dirname(__file__), 'train.py')
                else:
                    raise ValueError(f"Unknown command: {args.command}")
                
                jobs.append((script_path, command_args, args.debug))
    
    return jobs

def aggregate_results(output_dir):
    """결과 집계"""
    results_by_env = defaultdict(list)
    all_results = []
    
    # 모든 하위 디렉토리에서 결과 파일 찾기
    for root, dirs, files in os.walk(output_dir):
        if 'results.json' in files:
            with open(os.path.join(root, 'results.json'), 'r') as f:
                result = json.load(f)
                
                # 디렉토리 이름에서 정보 추출
                dir_name = os.path.basename(root)
                parts = dir_name.split('_')
                
                for part in parts:
                    if part.startswith('env'):
                        env_id = int(part[3:])
                        test_acc = result['test_accs'][0] if result['test_accs'] else 0
                        results_by_env[env_id].append(test_acc)
                        all_results.append({
                            'env': env_id,
                            'acc': test_acc,
                            'dir': root
                        })
    
    # 통계 계산
    summary = {}
    for env_id, accs in results_by_env.items():
        summary[f'env_{env_id}'] = {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'n': len(accs),
            'values': accs
        }
    
    # 전체 평균
    all_accs = [acc for accs in results_by_env.values() for acc in accs]
    if all_accs:
        summary['overall'] = {
            'mean': np.mean(all_accs),
            'std': np.std(all_accs),
            'n': len(all_accs)
        }
    
    return summary, all_results

def main():
    args = get_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 작업 목록 생성
    jobs = create_job_list(args)
    n_jobs = len(jobs)
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of jobs: {n_jobs}")
    print(f"Output directory: {args.output_dir}")
    
    if not args.skip_confirmation and not args.debug:
        response = input(f"\nProceed with {n_jobs} experiments? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # 실험 실행
    print(f"\nRunning {n_jobs} experiments...")
    
    for i, job in enumerate(jobs):
        print(f"\n[{i+1}/{n_jobs}] Running experiment...")
        result = run_experiment(job)
        
        if result['status'] == 'success':
            print(f"✓ Completed in {result['elapsed_time']:.1f} seconds")
        elif result['status'] == 'failed':
            print(f"✗ Failed: {result['error']}")
            if result['stderr']:
                print("Error output:", result['stderr'][-500:])  # 마지막 500자만
        elif result['status'] == 'debug':
            print(f"Debug mode - would run: {result['command']}")
    
    # 결과 집계
    if not args.debug:
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        
        summary, all_results = aggregate_results(args.output_dir)
        
        # 환경별 결과 출력
        for env_name, stats in sorted(summary.items()):
            if env_name != 'overall':
                print(f"\n{env_name}:")
                print(f"  Mean accuracy: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Number of runs: {stats['n']}")
        
        # 전체 평균 출력
        if 'overall' in summary:
            print(f"\nOverall:")
            print(f"  Mean accuracy: {summary['overall']['mean']:.4f} ± {summary['overall']['std']:.4f}")
            print(f"  Total runs: {summary['overall']['n']}")
        
        # 결과 저장
        summary_path = os.path.join(args.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
        
        # 최고 성능 모델 찾기
        if all_results:
            best_result = max(all_results, key=lambda x: x['acc'])
            print(f"\nBest model:")
            print(f"  Environment: {best_result['env']}")
            print(f"  Accuracy: {best_result['acc']:.4f}")
            print(f"  Directory: {best_result['dir']}")

if __name__ == '__main__':
    main()
