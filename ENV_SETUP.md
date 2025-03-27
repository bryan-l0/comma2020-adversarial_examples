This assumes one has Miniconda installed. It will probably work with regular Conda but I haven't tested it.

1. Create and load a new conda environment with `python=3.8`

        conda create -n adversarial python=3.8
        conda activate adversarial

2. Install a version of Pytorch that is < 2. E.g. for a SM86 gpu (RTX 30 series):

        pip install -r torch_sm86

More versions of Pytorch can be found here: [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)

3. Install the rest of the requirements. You might have to tinker with which version of things you have installed, but I have verified the ones in `requirements-sm86.txt` work on a 3070.

        pip install -r requirements-sm86.txt
