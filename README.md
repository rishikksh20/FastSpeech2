# fastspeech
FastSpeech

For training
```buildoutcfg
python3 train_fastspeech.py --outdir results
```

For inference
```buildoutcfg
python3 synthesis.py --out results --text "Germany is behind on cutting its carbon emissions so the lower house of Parliament passed." --path checkpoints\checkpoint_355k_steps.pyt
```