TEXT_PROMPT="cot_zh4"
PROMPT_FN="normal"
IMAGE_TOKEN_PROMPT="special_image_token"
MODEL="Llava-v1.5-13B"
OUTPUT="cot_zh/${MODEL}-${TEXT_PROMPT}.json"

srun python main.py -ds ${RPTS_PATH} \
                    -tp ${TEXT_PROMPT} \
                    -pf ${PROMPT_FN} \
                    -itp ${IMAGE_TOKEN_PROMPT} \
                    -m ${MODEL} \
                    -o ${OUTPUT} \
                    -l "zh" \
                    --fp16 