# 프로젝트 시작하기

- github 에서 클론

```bash
git clone https://github.com/SakanaAI/AI-Scientist.git
```

## 아나콘다 가상환경 시작 및 패키지 설치

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# Install pdflatex
sudo apt-get install texlive-full

# Install pypi requirements
pip install -r requirements.txt
```

- **textlive-full**, **window** 버전 댜운로드 : https://www.tug.org/texlive/windows.html

- 패키지 설치중 오류가 발생하면 아나콘다 가상머신을 삭제한 후 다시 시작한다
- 아나콘다 가상머신 폴더 : `C:\Users\USERNAME\anaconda3\envs`

### CUDA 11.8

```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
```

### CUDA 11.7

```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```

### CPU 버전 torch, CUDA 머신이 없을 경우 CPU 모드로 설정하기

```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
```

- 설치중 오류가 나면 다음 명령 실행후 다시 실행

```bash
pip3 uninstall aider-chat
pip3 uninstall filelock
pip3 uninstall networkx
pip3 uninstall pillow
pip3 uninstall fsspec

pip3 install aider-chat==0.53.0
pip3 install fsspec==2024.3.1
pip3 install pillow==10.4.0
pip3 install networkx==3.2.1
pip3 install filelock==3.15.4
```

## Setup NanoGPT

# Prepare NanoGPT data

```bash
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
```

## Create baseline runs (machine dependent)

- Set up NanoGPT baseline run
- 참고: 먼저 위의 준비 스크립트를 실행해야 합니다!

```bash
cd templates/nanoGPT
python experiment.py --out_dir run_0
python plot.py
```
