## Docker 에서 Sakana.ai 실행하기

## nVidia GPU 지원 이미지 실행 테스트
```bash
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```
## 3초마다 CUDA 실행상황 알아보기
```bash
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi -l 3
```

## 최신버전 Sakana Image 다운로드
```bash
docker pull t4646/ai-scientist:20240817
```

## endpoint Script
```bash
docker run -d --rm --gpus all --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment nanoGPT --num-ideas 2
```


```bash
docker run --gpus all --memory 10g --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment 2d_diffusion --num-ideas 2
```

```shell
docker run --gpus all --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment nanoGPT --num-ideas 2
```

```shell
docker run --gpus all --env-file=/app/ai-project/.env -v "/app/Ai-Scientist/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment nanoGPT --num-ideas 2
```

```shell
docker run -d --gpus all --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment nanoGPT_lite --num-ideas 2
```


```bash
docker run --gpus all --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment 2d_diffusion --num-ideas 1
```

```bash
docker run --gpus all --env-file=/c/app/ai-project/.env -v "/$(pwd)/templates:/c/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model chatgpt-4o-latest	 --experiment 2d_diffusion --num-ideas 1
```


```bash
docker pull nouranhamdy1998/ai-scientist
````
```bash
docker run --gpus all --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" nouranhamdy1998/ai-scientist --model gpt-4o-2024-05-13 --experiment 2d_diffusion --num-ideas 1
```


```bash
docker pull nouranhamdy1998/ai-scientist
```


## interactive
```bash
docker run --gpus all -it --env-file=/c/app/ai-project/.env --entrypoint /bin/bash t4646/ai-scientist:20240817
```

## docker 수정 후 실행중인 컨테이너 다른이름으로 이미지 복제 저장
```bash
docker commit d7d43aa61755 callor/ai-scientist
```