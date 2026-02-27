from lm_eval.__main__ import cli_evaluate
import outliers.model  # support hf-outliers model by importing into namespace

if __name__ == '__main__':
    cli_evaluate()