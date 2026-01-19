TEXT_PROMPT="cot_zh4"
PROMPT_FN="image_path"
IMAGE_TOKEN_PROMPT="without"
MODEL="Qwen-VL-Chat"
OUTPUT="cot_zh/${MODEL}-${TEXT_PROMPT}.json"

srun python main.py -ds ${RPTS_PATH} \
                    -tp ${TEXT_PROMPT} \
                    -pf ${PROMPT_FN} \
                    -itp ${IMAGE_TOKEN_PROMPT} \
                    -m ${MODEL} \
                    -o ${OUTPUT} \
                    -l "zh" \
                    --bf16