# RLHF\_instructGPT
Reproduce instructGPT

## Install
```
git clone git@github.com:LanXiu-0523/RLHF_instructGPT.git
cd RLHF_instructGPT

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
**1.单机单卡：**
```bash
bash train.sh sgl_gpu
```
**2.单机多卡：**
```bash
bash train.sh sgl_mach
```
**3.多机多卡**
```bash
# 首次运行
bash applications/scripts/mul_mach/apt-install.sh
```
```bash
bash train.sh mul_mach
```

## Acknowledgement
This project is built upon the codebase of [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples). Sincere thanks to Microsoft for their hard work!
