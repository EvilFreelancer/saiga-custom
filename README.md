# Saiga-Custom Project

Welcome to the `saiga-custom` project, a comprehensive collection of Jupyter notebooks specifically designed for
training large language models on datasets from the [Saiga (rulm)](https://github.com/IlyaGusev/rulm) project. This
repository is an essential resource for anyone looking to leverage the advanced capabilities of the Saiga datasets for
language model training.

Our notebooks are crafted to provide intuitive, step-by-step guidance for training state-of-the-art LoRA adapters for
different language models, ensuring that even those new to the field can successfully navigate the complexities of
language model training.

## Repository Contents

### Jupyter Notebooks

* [yarn_mistral_7b_128k.ipynb](./yarn_mistral_7b_128k.ipynb) - this notebook contains a script for training the
  [NousResearch/Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) model. This model, an
  advancement over the base [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), incorporates
  the [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) algorithm, enabling it to handle a
  context size of up to 128k tokens. The notebook provides a detailed and user-friendly guide for training the LoRA
  adapter specifically for the Yarn-Mistral-7b-128k model. It meticulously outlines the necessary steps and parameters
  required to optimize performance and achieve the best possible results with this enhanced model.
* [rugpt35_13b.ipynb](./rugpt35_13b.ipynb) - This notebook focuses on training
  the [ruGPT-3.5-13B](https://huggingface.co/ai-forever/ruGPT-3.5-13B) model, a powerful
  language model specifically tailored for understanding and generating Russian text. It guides users through creating a
  LoRA layer for model adaptation and subsequently performing a conversion to the GGML format for optimized deployment.
* [llama2_7b_yakovlev.ipynb](./llama2_7b_yakovlev.ipynb) - This notebook provides a detailed guide for training a
  Russian language model based on the [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
  model. The model is trained to imitate a historical figure
  named [Ivan Yakovlevich Yakovlev](https://en.wikipedia.org/wiki/Ivan_Yakovlev).
* [pavelgpt_7b_128k.ipynb](./pavelgpt_7b_128k.ipynb) - This notebook provides a detailed guide for training a Russian
  language model based on
  the [NousResearch/Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) model. It is able to
  generate text in Russian, answer questions, solve simple logical puzzles and simple math calculations. It is optimized
  for INSTRUCT mode and it works better if you give it system prompt and only one instruction (without history at all).

### Scripts

* [test_lora.py](./test_lora.py) - this script features a console-based chat interface and a Conversation class that
  maintains a message history. It is specifically adapted to function seamlessly with the Mistral model. The script
  demonstrates a practical application of the model, showcasing its conversational abilities and providing a template
  for further custom implementations.
* [test_gguf.py](./test_gguf.py) - this script features a console-based chat interface and a Conversation class that
  maintains a message history. If is adapted to work with the GGML format of models.

## Pretrained models

* [evilfreelancer/saiga_mistral_7b_128k_lora](https://huggingface.co/evilfreelancer/saiga_mistral_7b_128k_lora)
* [evilfreelancer/ruGPT-3.5-13B-lora](https://huggingface.co/evilfreelancer/ruGPT-3.5-13B-lora)
* [evilfreelancer/ruGPT-3.5-13B-ggml](https://huggingface.co/evilfreelancer/ruGPT-3.5-13B-ggml)
* [evilfreelancer/llama2_7b_gguf_yakovlev](https://huggingface.co/evilfreelancer/llama2_7b_gguf_yakovlev)
* [evilfreelancer/PavelGPT-7B-128K-v0.1-LoRA](https://huggingface.co/evilfreelancer/PavelGPT-7B-128K-v0.1-LoRA)

## Dependencies

The notebooks and scripts in this repository depend on specific libraries and frameworks. Ensure you have the latest
versions of these dependencies installed:

* Python 3.11
* Jupyter Lab
* PyTorch >= 2.1
* transformers >= 4.30
* flash-attn >= 2.3
* joblib >= 1.1

To install all dependencies just execute following command:

```shell
pip install -r requirements.txt
```

## Contribution

Contributions to the saiga-custom project are welcome. If you have suggestions for improvement or have developed
additional tools or scripts that could benefit the community, please feel free to submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to the [Saiga (rulm)](https://github.com/IlyaGusev/rulm) project and all the contributors who have made
this work possible. For more information about the Saiga project, please visit their GitHub repository.
