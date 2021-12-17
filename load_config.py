import os, yaml

YAML = os.environ.get('YHYDRA_CONFIG')

if False:
    # Define data
    config = {
        'MAX_N_FRAGMENTS': 200,
        'TOLERANCE_DALTON': 0.01,
    }

    # Write YAML file
    with io.open(YAML, 'w', encoding='utf8') as outfile:
        yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

# Read YAML file
with open(YAML, 'r') as stream:
    CONFIG = yaml.safe_load(stream)