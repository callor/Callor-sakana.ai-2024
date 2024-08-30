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

- conda create 오류 발생할 경우

```bash
conda create -n ai_scientist python=3.11
bash: C:\ProgramDatanaconda3\Scripts: No such file or directory
```

- 다음 명령으로 clean 실행 후 create

```bash
conda clean -i
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
tlmgr update --all --reinstall-forcibly-removed
```

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
