DATA_PATH=$1
waitk=${2:-3}

ARCH=sim_mt_transformer_iwslt_de_en
SAVEDIR=checkpoints/iwslt_ende/wait-${waitk}/

fairseq-train $DATA_PATH \
  --user-dir sim_mt/ --task sim_translation --wait-k $waitk \
  --source-lang en --target-lang de \
  --arch $ARCH --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 5e-04 --min-lr 1e-09 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-update 300000 \
  --max-tokens 6000 \
  --save-dir $SAVEDIR \
  --seed 1 \
  --restore-file checkpoint_last.pt \
  --update-freq 1 \
  --encoder-embed-dim 256 --decoder-embed-dim 256 \
  --attention-dropout 0.0 \
  --activation-dropout 0.0 \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --encoder-layers 6 --decoder-layers 6 | tee -a $SAVEDIR/log 2>&1
