# Getting started with Sakana.ai GPU version Windows project

- When running in Windows **Powershell**, many errors occur, so let's work in **git bash shell** after installing **git-scm**.
- Sakana.ai Official Site : https://sakana.ai/ai-scientist/

## Visual Studio Install

- Download and Install from https://visualstudio.microsoft.com/ko/downloads/
  Make sure to check and install the **C/C++ development environment** in the installation options.

## Creating a project environment

### Install a high-performance **GPU** graphics card that supports **CUDA** for model training.

- It is recommended to choose a graphics card with a TITAN or Quadro-class GPU installed.
- Find a CUDA-enabled graphics card : https://developer.nvidia.com/cuda-gpus
- You must select a graphics card with a **Compute Capability** of at least **7.x** or higher.
- Install the driver for your graphics card : https://www.nvidia.com/ko-kr/geforce/drivers/

### Installing nVIDIA GPU CUDA Software

- CUDA ToolKit Download : https://developer.nvidia.com/cuda-downloads
- Download 12.4.x from CUDA Archive : https://developer.nvidia.com/cuda-toolkit-archive  
  The latest version of CUDA is 12.6.x, but the version supported by torch is 12.4.x, so download 12.4.x from the Archive and install it.
- cuDNN Download : https://developer.nvidia.com/rdp/cudnn-archive
- Unzip the cuDNN download file and paste it into the `C: / Program Files / NVIDIA GPU Computing Toolkit / CUDA / v12.4` folder.

### Check installation of GPU CUDA compiler

```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:30:10_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

### Check GPU CUDA activation

```bash
nvidia-smi
```

![alt text](./images/image-3.png)

## Start a project

### Download project source code

- Clone From github.com

```bash
git clone https://github.com/SakanaAI/AI-Scientist.git
```

- Always update to the latest project before running the project (running launch_scientist.py).
- Run the following command in the **AI-Scientist** folder:

```bash
git pull
```

### Start Anaconda Virtual Environment and Install Dependencies Package

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# Install pypi requirements
pip3 install -r requirements.txt
```

- If the following error occurs when running `conda create`, run clean and then `conda create` again.

```bash
# Error Message
bash: C:\ProgramDatanaconda3\Scripts: No such file or directory
```

```bash
# Solution : Anaconda Environment Clean
conda clean -i
conda create -n ai_scientist python=3.11
```

- If a `CondaError` error occurs when running the `conda activate` command
  After running the `source` command, run `conda activate` again

```bash
CondaError: Run 'conda init' before 'conda activate'

# Shell Profile Preferences
conda init bash
source ~/.bash_profile
conda activate ai_scientist
```

### Activate Anaconda GPU

- Run this only if you have problems recognizing your GPU when running **launch_scientist**

```bash
# Install according to the CUDA version from the following commands.
# If you installed it in this document environment, the second command is
conda install cuda -c nvidia
conda install cuda -c nvidia/label/cuda-12.4
```

- Check the torch link for each CUDA version and execute the command :  
  https://pytorch.org/get-started/locally/
- Run this only if you have trouble running **launch_scientist**

```bash
# Install torch v 12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Install Tex Tool (PDF creation tool)

##### Install pdflatex for Ubuntu

- The following command is only executable on Ubuntu Linux. For Windows installation, see the following:

```bash
sudo apt-get install texlive-full
```

##### Download and install Windows version

- **textlive-full**, **window** version download : `https://www.tug.org/texlive/windows.html`
- You need to download and install it from the link above. It will take quite a while to install.

- If you see the following warning after installation, run the **update command** below.

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

- TexLive **Running the Update Command**

```bash
tlmgr update --all --reinstall-forcibly-removed
```

## Setup NanoGPT

### Prepare NanoGPT data

- Before training the model and generating documentation, you must first run the following script.

```bash
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
```

### Project Start : Model training and sample paper generation

### 프로젝트 실행전에 점검할 사항

- You must obtain an API key from openAI (paid) and the environment variable **OPENAI_API_KEY** must be set.
- You must register at https://www.semanticscholar.org/product/api and obtain an API key (free of charge).
  The issued API key must be set in the environment variable **S2_API_KEY**

```bash
conda activate ai_scientist

# Run the paper generation.

# If you use openAI's gpt-4o-xx
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT --num-ideas 2

# If you use claude-3-6-sonet-xx
python launch_scientist.py --model "claude-3-5-sonnet-20240620" --experiment nanoGPT_lite --num-ideas 2
```
