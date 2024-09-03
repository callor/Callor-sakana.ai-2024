## Docker 에서 Sakana.ai 실행하기

## nVidia GPU 지원 이미지 실행 테스트
```bash
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

## 최신버전 Sakana Image 다운로드
```bash
docker pull t4646/ai-scientist:20240816
```

## endpoint Script
```bash
docker run e --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment 2d_diffusion --num-ideas 1
```

```bash
docker run --env-file=/app/ai-project/.env -v "/$(pwd)/templates:/app/ai-project/AI-Scientist/templates" t4646/ai-scientist:20240817 --model gpt-4o-2024-05-13 --experiment 2d_diffusion --num-ideas 1
```

## interactive
```bash
docker run -it --env-file=/app/ai-project/.env --entrypoint /bin/bash t4646/ai-scientist:20240817
```

## docker 수정 후 실행중인 컨테이너 다른이름으로 이미지 복제 저장
```bash
docker commit d7d43aa61755 callor/ai-scientist
```