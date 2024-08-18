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

## AIMET for Simulated Quantized Model
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

## [Optional] integer
If you'd like to try out the real quantized llama model, please install [integer](https://github.com/fwtan/integer).
