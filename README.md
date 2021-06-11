This repo contains codes for simultanous translation task for our paper *Temporally Correlated Task Scheduling for Sequence Learning* published on ICML 2021.
Codes for stock price forecasting task are at [https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TCTS](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TCTS).

Codes for simultaneous translation are in `sim_mt/`, and codes for our method are in `ours/`. Our code is based on [fairseq](https://github.com/pytorch/fairseq/) v0.8.0; please install it by cloning and installing their github repo, or by

`pip install fairseq==v0.8.0`

## Data preparation

For IWSLT'14 En-De, please prepare the data using fairseq's [script](https://github.com/pytorch/fairseq/blob/v0.8.0/examples/translation/prepare-iwslt14.sh),
and then binarize the data by running:

```bash
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en
```

For IWSLT'15 En-Vi, please download the [tokenized data](https://nlp.stanford.edu/projects/nmt/),
and then prepare the binarized data by running:

```bash
fairseq-preprocess --source-lang en --target-lang vi \
    --trainpref data/train --validpref data/tst2012 --testpref data/tst2013 \
    --destdir data/bins --workers 10 \
    --thresholdsrc 5 --thresholdtgt 5
```

## Training wait-k

For IWSLT'14 En-De, run training with `scripts/sim_mt/train_iwslt_ende.sh`:

```bash
bash scripts/sim_mt/train_iwslt_ende.sh $DATA_PATH $k
```

where `$DATA_PATH` is the binarized data and `k` is the training threshold.

Similarly, for IWSLT'15 En-Vi, run training with `scripts/sim_mt/train_iwslt_envi.sh`:

```bash
bash scripts/sim_mt/train_iwslt_envi.sh $DATA_PATH $k
```

## Inference wait-k

Run inference with `scripts/test/test.sh`:

```bash
bash scripts/test/test.sh $DATA_PATH $refF $cktpath $k $src $tgt
```

where
* `$DATA_PATH` is the binarized data
* `$refF` is the reference file
  * For IWSLT'14 En-De, `iwslt14.tokenized.de-en/tmp/test.de`
  * For IWSLT'15 En-Vi, `data/tst2013.vi`
* `$cktpath` is the checkpoint file to be evaluated
* `$k` is the waiting threshold
* `$src` and `$tgt` are the source and target languages (`en` and `de` for IWSLT'14 En-De, `en` and `vi` for IWSLT'15 En-Vi)

## Training Ours

Similar to training wait-k, for IWSLT'14 En-De, run training with `scripts/ours/train_iwslt_ende.sh`:

```bash
bash scripts/sim_mt/train_iwslt_ende.sh $DATA_PATH $k
```

For IWSLT'15 En-Vi, run training with `scripts/sim_mt/train_iwslt_envi.sh`:

```bash
bash scripts/sim_mt/train_iwslt_envi.sh $DATA_PATH $k
```

## Other baselines

1. Random: run the wait-k script, but use the following `fairseq-train` options: `--wait-k uniform`
2. CL: run the wait-k script, but use the following `fairseq-train` options: `--wait-k CL-linear --wait-k-sample-end $k`, where `$k` is the waiting threshold during inference.
3. WIW and WID: run evaluation with `scripts/test/test_agent.sh`:

```bash
bash scripts/test/test_agent.sh $DATA_PATH $refF $cktpath $agent $src $tgt
```

where `$agent` is `wait-if-worse` (WIW) or `wait-if-diff` (WID).
