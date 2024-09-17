# Roteiros de comandos para o workshop de `instructlab`

Aqui você vai encontrar roteiros para instalar e usar o instruct lab nas plataformas Linux+CUDA e Mac com hardware M1 ou M2.
Ainda há diferenças na sequência de comando nas duas plataformas por causa dos diferentes backends que são usados para inferência e treinamento nesses diferentes hardwares. No final o resultado será o mesmo, mas alguns detalhes vão variar.

## Roteiro Linux com CUDA

```
pip install torch scikit-build-core 'numpy<2.0.0' cmake  wheel
CUDACXX=/usr/local/cuda-12/bin/nvcc  CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CUDA=on -DLLAMA_NATIVE=off" FORCE_CMAKE=1   MAX_JOBS=4 pip install 'instructlab[cuda]'   --no-build-isolation
$ ilab system info
sys.version: 3.11.6 (main, Nov 26 2023, 21:29:42) [GCC 13.2.1 20231011 (Red Hat 13.2.1-4)]
sys.platform: linux
os.name: posix
platform.release: 6.10.5-100.fc39.x86_64
platform.machine: x86_64
platform.node: fedora
platform.python_version: 3.11.6
os-release.ID: fedora
os-release.VERSION_ID: 39
os-release.PRETTY_NAME: Fedora Linux 39 (Workstation Edition)
instructlab.version: 0.18.4
instructlab-dolomite.version: 0.1.1
instructlab-eval.version: 0.1.2
instructlab-quantize.version: 0.1.0
instructlab-schema.version: 0.3.1
instructlab-sdg.version: 0.2.7
instructlab-training.version: 0.4.2
torch.version: 2.3.1+cu121
torch.backends.cpu.capability: AVX512
torch.version.cuda: 12.1
torch.version.hip: None
torch.cuda.available: True
torch.backends.cuda.is_built: True
torch.backends.mps.is_built: False
torch.backends.mps.is_available: False
torch.cuda.bf16: True
torch.cuda.current.device: 0
torch.cuda.0.name: NVIDIA GeForce RTX 4060 Ti
torch.cuda.0.free: 14.7 GB
torch.cuda.0.total: 15.6 GB
torch.cuda.0.capability: 8.9 (see https://developer.nvidia.com/cuda-gpus#compute)
llama_cpp_python.version: 0.2.79
llama_cpp_python.supports_gpu_offload: True

# Opcional, instalação do vLLM
CUDACXX=/usr/local/cuda/bin/nvcc  CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install vllm --no-build-isolation
git clone https://github.com/maxdebayser/taxonomy.git
$ ilab config init

# Em outro terminal
# Alternativa 1
$ ilab model serve
# Alternativa 2 - modelo já baixado, mas muito lento para geração de dados sintéticos
$ vllm serve ~/.cache/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf --enforce-eager --chat-template chat_template.jinja
# Alternativa 3 - rápido, mas requer baixar o modelo não quantizado
# vllm serve instructlab/merlinite-7b-lab --quantization fp8 --max-model-len 8192

$ ilab model chat
$ ilab taxonomy diff  --taxonomy-path taxonomy/ --taxonomy-base=c14b55d
knowledge/cooking/brazilian/crepioca/qna.yaml
Taxonomy in taxonomy/ is valid :)
# Alternativa 1: geração com ilab model serve 25m min numa NVIDIA GeForce RTX 4060 Ti
$ time ilab data generate --pipeline simple  --taxonomy-path taxonomy/ --taxonomy-base=c14b55d
# Alternativa 2: geração com vllm 4 min numa NVIDIA GeForce RTX 4060 Ti
$ time ilab data generate  --pipeline simple  --taxonomy-path taxonomy/ --taxonomy-base=c14b55d --model instructlab/merlinite-7b-lab --endpoint-url http://localhost:8000/v1

# Treinamento sem quantização (precisa de muita memória)
ilab model train --legacy --device=cuda --gpus 1 --model-path instructlab/merlinite-7b-lab --input-dir ~/.local/share/instructlab/datasets
# Treinamento com quantização
ilab model train --legacy --device=cuda --gpus 1 --4-bit-quant --model-path instructlab/merlinite-7b-lab --input-dir ~/.local/share/instructlab/datasets

# Exemplo de saída
LINUX_TRAIN.PY: FINISHED
Copied training_results/checkpoint-1033/added_tokens.json to training_results/final
Copied training_results/checkpoint-1033/special_tokens_map.json to training_results/final
Copied training_results/checkpoint-1033/tokenizer.json to training_results/final
Copied training_results/checkpoint-1033/tokenizer.model to training_results/final
Copied training_results/checkpoint-1033/tokenizer_config.json to training_results/final
Copied training_results/merged_model/config.json to training_results/final
Copied training_results/merged_model/generation_config.json to training_results/final
Moved training_results/merged_model/model.safetensors to training_results/final
SKIPPING CONVERSION to gguf. This is unsupported with --4-bit-quant. See https://github.com/instructlab/instructlab/issues/579.


# Num outro terminal
$ vllm serve instructlab/merlinite-7b-lab --quantization fp8 --enable-lora --lora-modules crepioca=training_results/checkpoint-1033 --max-model-len 2048

$ ilab model chat --model crepioca -gm -qq "Como fazer crepioca?"
```

Chat template para usar com modelo GGUF no vllm
```
{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>'+ '\n' + message['content'] + '\n'}}{% elif message['role'] == 'user' %}{{'<|user|>' + '\n' + message['content'] + '\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n')}}{% endif %}{% endfor %}
```


## Roteiro mac
```
$ ilab system info
Verificar -> llama_cpp_python.supports_gpu_offload: True
$ git clone https://github.com/maxdebayser/taxonomy.git
$ ilab config init
$ ilab model download

$ ilab model serve # em outro terminal
$ ilab model chat -gm --q "Como fazer crepioca?"
A crepioca é um delicioso e tradicional doce português, semelhante à uma torta de ovos com um recheio de leite condensado e canela. Para preparar essa delícia em casa, siga as instruções abaixo:
$ ilab taxonomy diff  --taxonomy-path taxonomy/ --taxonomy-base=c14b55d
knowledge/cooking/brazilian/crepioca/qna.yaml
Taxonomy in taxonomy/ is valid :)
$ time ilab data generate --pipeline simple  --taxonomy-path taxonomy/ --taxonomy-base=c14b55d

Exemplo de saída:
...
Creating json from Arrow format: 100%|_________________________________________________________________________________________________________________________________________________________________________| 2/2 [00:00<00:00, 17.93ba/s]
INFO 2024-09-16 23:37:04,991 instructlab.sdg.datamixing:200: Mixed Dataset saved to /Users/m1/Library/Application Support/instructlab/datasets/skills_train_msgs_2024-09-16T22_43_36.jsonl
INFO 2024-09-16 23:37:04,992 instructlab.sdg:438: Generation took 3208.78s
ilab data generate --pipeline simple --taxonomy-path taxonomy/   5.09s user 2.08s system 0% cpu 53:30.26 total


# Treinamento - Aqui tem que mater o ilab model serve no outro terminal
#
$ ilab model train --model-path instructlab/merlinite-7b-lab --input-dir "${HOME}/Library/Application Support/instructlab/datasets"

# Treinamento com modelo quantizado (já baixado no início)
ilab model train --gguf-model-path "${HOME}/Library/Caches/instructlab/models/merlinite-7b-lab-Q4_K_M.gguf" --tokenizer-dir instructlab/merlinite-7b-lab --model-path instructlab/merlinite-7b-lab  --input-dir "${HOME}/Library/Application Support/instructlab/datasets"

$ ilab model convert --model-dir  "${HOME}/Library/Application\ Support/instructlab/checkpoints/instructlab-merlinite-7b-lab-mlx-q"
$ ilab model serve --model-path instructlab-merlinite-7b-lab-trained/instructlab-merlinite-7b-lab-Q4_K_M.gguf
$ ilab model chat -gm -qq "Como fazer crepioca?"
The response would be: 'To make crepioca, you need one tablespoon of tapioca flour mixed with two tablespoons of water and one egg.
```
