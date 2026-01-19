TEXT_PROMPT="cot_zh1"
PROMPT_FN="normal"
IMAGE_TOKEN_PROMPT="special_image_token"
MODEL="Llava-Next-34B"
OUTPUT="cot_zh/${MODEL}-${TEXT_PROMPT}.json"

srun python main.py -ds ${RPTS_PATH} \
                    -tp ${TEXT_PROMPT} \
                    -pf ${PROMPT_FN} \
                    -itp ${IMAGE_TOKEN_PROMPT} \
                    -m ${MODEL} \
                    -o ${OUTPUT} \
                    -l "zh" \
                    --nf4 \
                    --fp16 