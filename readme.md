# Llama3 from scratch with MLX

This is a nanoGPT-like implementation of llama3 architecture model from scratch using apple's `mlx` python library. Most of the things are implemented from scratch and no `nn` module used. It will also work if you just swap in `numpy`.

## Setup
1. Install packages.

All we need is just `mlx` for our model.
``` bash
pip install mlx
```
However, we will need these packages for converting pytorch weight and loading tokenization

``` bash
pip install numpy torch llama_models
```
2. Download the model. I have only tested the `Llama3.2-1B`. You can download it from https://www.llama.com/llama-downloads/

3. Update the weight, param and tokenizer paths in `main.py` to your download destination.

## Running
``` bash
python main.py
```
You can directly specify your prompt, temperature, topk, etc. in `main.py`

`playground.ipynb` is just me studying and experimenting with the model components. It's kind of documenting the learning journey, and I think will be helpful for somebody who is trying to start doing something similar. So I committed it here.
## Special thanks to
Architecture understanding
- [Graham Neubig's CMU NLP course](https://youtu.be/QkGwxtALTLU?si=jYu-qasbIteDUUZo)
- [3b1b's DL series](https://youtu.be/9-Jl0dxWQs8?si=H1b-F_GiVMW5ZWhV)

Implementation references
- https://github.com/meta-llama/llama3/blob/main/llama/model.py
- https://gist.github.com/yacineMTB/8ea60050a18e0b613f3dfd0cb4955625
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L286
