# 环境变量配置
export WORKDIR="/cfs/hadoop-mlp-ckpt/gnmodel/code/megatron-deepspeed"
export PYTHONPATH="/cfs/hadoop-mlp-ckpt/gnmodel/code/megatron-deepspeed"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_LIBRARY_PATH=/usr/local/conda/lib/python3.9/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
export PATH="$PATH:/usr/local/conda/bin"
export NCCL_DEBUG=WARN
export NCCL_PXN_DISABLE=1
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_TIMEOUT=22

# shellcheck disable=SC2164
cd $WORKDIR

# 配置文件及机器信息读取
export MLP_MPI_HOSTFILE="${WORKDIR}/hostfile.txt"
python3 hope_gen_hostfile.py --target_path $MLP_MPI_HOSTFILE

DATA_DIR="/cfs/hadoop-mlp-ckpt/gndata"
DATA_PATH="$DATA_DIR/en/redpajama/v1/arxiv/tokenized_text_document"


#DATA_PATH="
#  0.024491595300879247 $DATA_DIR/en/redpajama/v1/arxiv/tokenized_text_document \
#  0.03918655248140679 $DATA_DIR/en/redpajama/v1/book/tokenized_text_document \
#  0.08278159211697185 $DATA_DIR/en/redpajama/v1/c4/tokenized_text_document \
#  0.0685764668424618 $DATA_DIR/en/redpajama/v1/common_craw_merge/2019-30-part/tokenized_text_document \
#  0.0685764668424618 $DATA_DIR/en/redpajama/v1/common_craw_merge/2020-05-part/tokenized_text_document \
#  0.0685764668424618 $DATA_DIR/en/redpajama/v1/common_craw_merge/2022-05-part/tokenized_text_document \
#  0.0685764668424618 $DATA_DIR/en/redpajama/v1/common_craw_merge/2023-06-part-v2/tokenized_text_document \
#  0.0007200529018458498 $DATA_DIR/en/redpajama/v1/stackexchange/tokenized_text_document \
#  0.00783731049628136 $DATA_DIR/en/wiki/tokenized_text_document \
#  0.006073915634618053 $DATA_DIR/en/pile/Wikipedia/tokenized_text_document \
#  0.0041145880105477135 $DATA_DIR/en/pile/DMMathematics/tokenized_text_document \
#  0.00023511931488844076 $DATA_DIR/en/pile/EnronEmails/tokenized_text_document \
#  0.0013225461462474796 $DATA_DIR/en/pile/EuroParl/tokenized_text_document \
#  0.009306806214334114 $DATA_DIR/en/pile/FreeLaw/tokenized_text_document \
#  0.005877982872211019 $DATA_DIR/en/pile/Gutenberg/tokenized_text_document \
#  0.0009649688548546423 $DATA_DIR/en/pile/HackerNews/tokenized_text_document \
#  0.0003379840151521336 $DATA_DIR/en/pile/NIHExPorter/tokenized_text_document \
#  0.0025961091018932 $DATA_DIR/en/pile/OpenSubtitles/tokenized_text_document \
#  0.044084871541582644 $DATA_DIR/en/pile/OpenWebText2/tokenized_text_document \
#  0.000587798287221102 $DATA_DIR/en/pile/PhilPapers/tokenized_text_document \
#  0.0038696720575389213 $DATA_DIR/en/pile/PubMedAbstracts/tokenized_text_document \
#  0.021062771958756152 $DATA_DIR/en/pile/PubMedCentral/tokenized_text_document \
#  0.000489831906017585 $DATA_DIR/en/pile/UbuntuIRC/tokenized_text_document \
#  0.001175596574442204 $DATA_DIR/en/pile/YoutubeSubtitles/tokenized_text_document \
#  0.0157331655129012 $DATA_DIR/zh/wudao/v3_tokenized/tokenized_text_document \
#  0.13373190685966022 $DATA_DIR/zh/cc/v2_exact_dedup_merged/tokenized \
#  0.003697293895531782 $DATA_DIR/zh/zhihu_v2/zhihu_qa_without_url/tokenized_text_document \
#  0.005624606670862179 $DATA_DIR/zh/zhihu_v2/zhihu_article_without_url/tokenized_text_document \
#  0.0235997482693518 $DATA_DIR/zh/gzh_merged/tokenized_text_document \
#  0.0471994965387036 $DATA_DIR/zh/rubish/tokenized/_text_document \
#  0.0235997482693518 $DATA_DIR/zh2/fudan_book/v1/cbooks_epub_mobi_merge/tokenized_text_document \
#  0.0078665827564506 $DATA_DIR/en/bk/baike_clean_v2/tokenized_text_document \
#  0.002910635619886722 $DATA_DIR/en/bk/baike_in_zhidao_clean_v2/tokenized_text_document \
#  0.0009203901825047201 $DATA_DIR/zh_en_translation/tokenized_text_document \
#  0.0008967904342353684 $DATA_DIR/reverse_zh_en_translation/tokenized_text_document \
#  0.00039332913782253 $DATA_DIR/zh2/shiti/output/20230722/tokenized_text_document \
#  0.03382630585273758 $DATA_DIR/zh/zhidao/stage2_tokenized/tokenized_text_document \
#  0.09957834793297136 $DATA_DIR/code/starcoderdata/tokenized_text_document \
#  1.5559116864526773e-06 $DATA_DIR/en/math1/tokenized_text_document \
#  0.00042009615534222293 $DATA_DIR/en/math2/tokenized_text_document \
#"

SEQ_LEN=4096

#MODEL_SIZE=33
#NUM_LAYERS=40
#HIDDEN_SIZE=6656
#NUM_ATTN_HEADS=52
#GLOBAL_BATCH_SIZE=2048

MODEL_SIZE=1
NUM_LAYERS=22
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
GLOBAL_BATCH_SIZE=2048

###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
## For MoE model, we found sometimes training a bit more to 330B tokens helps
#TRAIN_TOKENS=3200000000000
TRAIN_TOKENS=32000000

## TRAIN_ITERS is another termination condition and also affect the number of
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some steps,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_ITERS.
TRAIN_ITERS=$(( ${TRAIN_TOKENS} * 3 / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
## For MoE model, we found that setting the decay token to 300B helps.
WARMUP_TOKENS=37500 # 看看经验值
# LR_DECAY_TOKENS=260000000000
LR_DECAY_TOKENS=3000000 # 看看经验值
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*TP_SIZE/NUM_GPUS
BATCH_SIZE=4

## Model parallelism, 1 is no MP
TP_SIZE=2

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=1

NUM_GPUS_PERNODE=8
NUM_NODE=1
NUM_GPUS=8
###############################################################################
### MoE configs
## Number of experts. EP_SIZE 1 means dense model without MoE
EP_SIZE=1
#EP_SIZE=16

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi

## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
LR=4.5e-4 # 看看经验值
MIN_LR=4.5e-06

## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01 # 后做对比

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
GATE_TOPK=2
MOE_DROP_TOKEN="false" # 主要看 mfu，最终 loss 也要看一下
###############################################################################
### Misc configs
LOG_INTERVAL=1
EVAL_ITERS=100
EVAL_INTERVAL=100
SAVE_INTERVAL=100

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
#INIT_STD=0.014
INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
ACTIVATION_CHECKPOINT="true"
# ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${TP_SIZE}-pp-${PP_SIZE}"
if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

OUTPUT_BASEPATH=$WORKDIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

VOCAB_PATH="/cfs/hadoop-mlp-ckpt/gnmodel/vocab/spiece.model"
###############################################################################
data_options=" \
         --tokenizer-type LightyearTokenizer
         --vocab-file ${VOCAB_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"

megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${TP_SIZE} \
        --use-flash-attn \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --rotary-percent 1.0 \
        --disable-bias-linear \
        --untie-embeddings-and-output-weights \
        --bf16 \
        --no-masked-softmax-fusion \
        --swiglu \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --topk ${GATE_TOPK} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --load ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR}"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

template_json=$WORKDIR/examples_deepspeed/MoE/"exp_ds_config_gpt.json"
config_json="ds_config_gpt_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/0/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
	  > ${config_json}

deepspeed_options=" \
		    --deepspeed \
		    --deepspeed_config ${config_json} \
		    --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
ITERATION_FILE="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
ITERATION_FILE_2="$CHECKPOINT_PATH/latest"
ITERATION=0
for (( node = 0; node <= NUM_NODE-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$ITERATION_FILE\""); then
        LOCAL_ITERATION=$(ssh -q worker-"$node" cat $ITERATION_FILE)
        ITERATION=$(( ${LOCAL_ITERATION} > ${ITERATION} ? ${LOCAL_ITERATION} :  ${ITERATION} ))
    fi
done
if [[ $ITERATION -gt 0 ]]; then
    ITERATION_2="global_step${ITERATION}"
    ds_ssh "echo $ITERATION > $ITERATION_FILE"
    ds_ssh "echo $ITERATION_2 > $ITERATION_FILE_2"
fi

#run_cmd="deepspeed --hostfile=$MLP_MPI_HOSTFILE $WORKDIR/pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
run_cmd="deepspeed  $WORKDIR/pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}"
echo ${run_cmd}
eval ${run_cmd}
set +x

