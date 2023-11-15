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

### Scripts

* [test_lora.py](./test_lora.py) - this script features a console-based chat interface and a Conversation class that
  maintains a message history. It is specifically adapted to function seamlessly with the Mistral model. The script
  demonstrates a practical application of the model, showcasing its conversational abilities and providing a template
  for further custom implementations.

## Pretrained models

* [evilfreelancer/saiga_mistral_7b_128k_lora](https://huggingface.co/evilfreelancer/saiga_mistral_7b_128k_lora)

## Dependencies

The notebooks and scripts in this repository depend on specific libraries and frameworks. Ensure you have the latest
versions of these dependencies installed:

* Python 3.11
* Jupyter Lab
* PyTorch >= 2.0
* transformers>=4.30
* flash-attn>=2.3

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
