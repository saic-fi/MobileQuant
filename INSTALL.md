# Install packages 

## Python Environment

```
# lm-eval-harness requires python >= 3.9
conda create -n llm python=3.10 
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
conda install pkg-config cmake eigen==3.3.7 
conda install -c conda-forge liblapacke
```

## Qualcomm AIMET, required only for the on-device deployment
```
git clone https://github.com/fwtan/aimet.git
cd aimet
git checkout user/fuwen.tan/main
mkdir build
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_CUDA=ON -DENABLE_TORCH=ON -DENABLE_TENSORFLOW=OFF -DENABLE_ONNX=OFF -DENABLE_TESTS=OFF
make -j8
make install
```

Please include the path in your ```.bashrc```:

```
export PYTHONPATH=${AIMET_ROOT}/build/staging/universal/lib/python:${PYTHONPATH}
```

## Qualcomm AI Engine Direct SDK (QNN), required only for the on-device deployment
Please follow the instruction from https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html to install QNN, as well as the required packages (e.g. Android NDK, etc).
The code has been tested with QNN 2.22.6.240515
