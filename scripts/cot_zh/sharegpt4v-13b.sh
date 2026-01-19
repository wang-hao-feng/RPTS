TEXT_PROMPT="cot_zh2"
PROMPT_FN="normal"
IMAGE_TOKEN_PROMPT="special_image_token"
MODEL="ShareGPT4V-13B"
OUTPUT="cot_zh/${MODEL}-${TEXT_PROMPT}.json"

srun python main.py -ds ${RPTS_PATH} \
                    -tp ${TEXT_PROMPT} \
                    -pf ${PROMPT_FN} \
                    -itp ${IMAGE_TOKEN_PROMPT} \
                    -m ${MODEL} \
                    -o ${OUTPUT} \
                    -l "zh" \
                    --fp16