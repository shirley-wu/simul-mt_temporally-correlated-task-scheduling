DATA_PATH=$1
refF=$2
cktpath=$3
waitk=$4
srclng=${5:-en}
tgtlng=${6:-de}

subset="test"
beam=5
lenpen=1.0
bs=1024

tgtlog=$(dirname $cktpath)/${subset}/$(basename $cktpath).wait_$waitk.log

fairseq-generate $DATA_PATH --gen-subset $subset --path $cktpath \
    --user-dir sim_mt/ --task sim_translation --wait-k $waitk --print-alignment \
    --batch-size $bs --beam $beam --lenpen $lenpen -s ${srclng} -t ${tgtlng} > $tgtlog

python scripts/test/cut_generated_results.py $tgtlog $tgtlog.sys.bpe
cat $tgtlog.sys.bpe | sed "s/@@ //g" > $tgtlog.sys

BLEUer="/tmp/mosesdecoder/scripts/generic/multi-bleu.perl"
if [ ! -f $BLEUer ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
perl $BLEUer $refF < $tgtlog.sys | tee $tgtlog.latency.bleu

python scripts/test/calc_latency_from_log.py $tgtlog $waitk | tee -a $tgtlog.latency.bleu
