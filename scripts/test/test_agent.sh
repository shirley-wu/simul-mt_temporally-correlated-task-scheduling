DATA_PATH=$1
refF=$2
cktpath=$3
agent=$4  # wait-if-worse or wait-if-diff
srclng=${5:-en}
tgtlng=${6:-de}

subset="test"
beam=5
lenpen=1.0
bs=1024

tgtlog=$(dirname $cktpath)/${subset}/$(basename $cktpath).agent_${agent}.log

fairseq-generate $DATA_PATH --gen-subset $subset --path $cktpath \
    --left-pad-source False --left-pad-target False \
    --agent $agent --rw-output $tgtlog.rw \
    --user-dir sim_mt/ --task agent_sim_translation \
    --batch-size $bs --beam $beam --lenpen $lenpen -s $srclng -t $tgtlng > $tgtlog

python scripts/test/cut_generated_results.py $tgtlog $tgtlog.sys.bpe
cat $tgtlog.sys.bpe | sed "s/@@ //g" > $tgtlog.sys

BLEUer="/tmp/mosesdecoder/scripts/generic/multi-bleu.perl"
if [ ! -f $BLEUer ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
perl $BLEUer $refF < $tgtlog.sys | tee $tgtlog.latency.bleu

python scripts/test/calc_latency_from_log.py $tgtlog $waitk | tee -a $tgtlog.latency.bleu
