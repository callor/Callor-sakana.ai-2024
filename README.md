# 프로젝트 시작하기

- 윈도우 powershell 에서 실행할 경우 많은 오류가 발생 `git-scm` 설치한 후 `git bash shell` 에서 작업

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

- bash 에서 activate 오류 발생하면 `source` 명령 실행 후 `conda activate`

```bash
CondaError: Run 'conda init' before 'conda activate'
```

```bash
source ~/.bash_profile
```

- `sudo apt-get install textlive-full` 명령은 **Ubuntu** Linux 명령으로 윈도우에서는 다음의 링크에서 다운로로드 받아 설치해야 한다. 설치하는 시간이 상당히 오래 걸린다
- **textlive-full**, **window** 버전 댜운로드 : https://www.tug.org/texlive/windows.html
- 설치 후 다음과 같은 경고가 나오면 업데이트 실행

```bash
*** PLEASE READ THIS WARNING ***********************************

The following (inessential) packages failed to install properly:

  tex4ht

You can fix this by running this command:

to complete the installation.

However, if the problem was a failure to download (by far the
most common cause), check that you can connect to the chosen mirror
in a browser; you may need to specify a mirror explicitly.
******************************************************************
```

```bash
tlmgr update --all --reinstall-forcibly-removed
```

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

## Setup NanoGPT

### Prepare NanoGPT data

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

### CPU 버전 torch, CUDA 머신이 없을 경우 CPU 모드로 설정하기

```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
```

- conda 로 설치하기

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### CUDA GPU 없는 경우 실행오류 발생, CPU 버전으로 변경

- CPU 버전으로 실행할 경우 모델학습 단계에서 문제가 발생한다.

```bash
Error loading "C:\ProgramData\anaconda3\envs\ai_scientist\Lib\site-packages\torch\lib\fbgemm.dll
```

- CPU 버전으로 실행하기 위한 pyTorch 설치 : 모두 실패

```bash
pip3 uninstall torch torchvision torchaudio
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
