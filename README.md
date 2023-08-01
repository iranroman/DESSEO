# DESSEO: dynamic and elastic synchronization with speech envelope onsets

To install:
```
git clone https://github.com/iranroman/DESSEO
cd DESSEO
pip install -e .
```

To train the model run:
```bash
python tools/run.py --config config/Golumbic_data.yaml          
```

To run inference and generate plots with model results run:
```bash
python tools/run.py --config config/Golumbic_data_inferene.yaml          
```
