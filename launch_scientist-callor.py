import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex

# 리플렉션 횟수 설정
NUM_REFLECTIONS = 3

# 현재 시간을 출력하는 유틸리티 함수
def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 명령줄 인자 파싱 함수
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    
    # 아이디어 생성 및 검사 관련 인자
    parser.add_argument("--skip-idea-generation", action="store_true", help="아이디어 생성을 건너뛰고 기존 아이디어 로드")
    parser.add_argument("--skip-novelty-check", action="store_true", help="새로운 아이디어 검사 건너뛰기")
    
    # 실험과 모델 설정
    parser.add_argument("--experiment", type=str, default="nanoGPT", help="실험 이름 설정")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620", choices=AVAILABLE_LLMS, help="사용할 모델 선택")
    
    # 보고서 작성 형식 설정
    parser.add_argument("--writeup", type=str, default="latex", choices=["latex"], help="보고서 형식 설정")
    
    # 병렬 프로세스 및 GPU 설정
    parser.add_argument("--parallel", type=int, default=0, help="병렬 실행 프로세스 수. 0이면 순차 실행")
    parser.add_argument("--improvement", action="store_true", help="리뷰 기반 개선 활성화")
    parser.add_argument("--gpus", type=str, default=None, help="사용할 GPU ID 리스트")
    
    # 아이디어 생성 수 설정
    parser.add_argument("--num-ideas", type=int, default=50, help="생성할 아이디어 수")
    
    return parser.parse_args()

# 사용 가능한 GPU 확인 함수
def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))

# 작업자 함수: 병렬 실행 시 사용
def worker(queue, base_dir, results_dir, model, client, client_model, writeup, improvement, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} 시작.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=True
        )
        print(f"완료된 아이디어: {idea['Name']}, 성공 여부: {success}")
    print(f"Worker {gpu_id} 종료.")

# 아이디어 수행 함수
def do_idea(base_dir, results_dir, idea, model, client, client_model, writeup, improvement, log_file=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"폴더 {folder_name} 이미 존재."
    shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

    # 기본 결과 로드
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}

    exp_file = osp.join(folder_name, "experiment.py")
    notes = osp.join(folder_name, "notes.txt")

    # 로그 설정
    if log_file:
        original_stdout = sys.stdout
        log = open(osp.join(folder_name, "log.txt"), "a")
        sys.stdout = log

    try:
        print_time()
        print(f"*아이디어 시작: {idea_name}*")
        
        # 실험 수행
        success = perform_experiments(idea, folder_name, Coder(model), baseline_results)
        if not success:
            print(f"실험 실패: {idea_name}")
            return False

        # 보고서 작성
        if writeup == "latex":
            perform_writeup(idea, folder_name, Coder(model), client, client_model)
            print("보고서 작성 완료.")

        # 리뷰 수행
        paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
        review = perform_review(paper_text, model="gpt-4o-2024-05-13", client=openai.OpenAI())
        with open(osp.join(folder_name, "review.txt"), "w") as f:
            f.write(json.dumps(review, indent=4))

        # 개선 수행
        if improvement:
            perform_improvement(review, Coder(model))
            print("개선 완료.")

        return True
    except Exception as e:
        print(f"아이디어 평가 실패: {idea_name}, 오류: {str(e)}")
        return False
    finally:
        if log_file:
            sys.stdout = original_stdout
            log.close()

if __name__ == "__main__":
    args = parse_arguments()

    # GPU 확인 및 병렬 프로세스 설정
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(f"경고: 요청된 병렬 수 {args.parallel}가 사용 가능한 GPU 수 {len(available_gpus)}를 초과하여 조정합니다.")
        args.parallel = len(available_gpus)

    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    # 아이디어 생성 및 평가 수행
    ideas = generate_ideas(base_dir, client=client, model=client_model, max_num_generations=args.num_ideas)
    ideas = check_idea_novelty(ideas, base_dir=base_dir, client=client, model=client_model)

    if args.parallel > 0:
        queue = multiprocessing.Queue()
        for idea in ideas:
            queue.put(idea)
        processes = [
            multiprocessing.Process(
                target=worker, args=(queue, base_dir, results_dir, args.model, client, client_model, args.writeup, args.improvement, gpu_id)
            )
            for gpu_id in available_gpus
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    else:
        for idea in ideas:
            do_idea(base_dir, results_dir, idea, args.model, client, client_model, args.writeup, args.improvement)
    print("모든 아이디어 평가 완료.")
