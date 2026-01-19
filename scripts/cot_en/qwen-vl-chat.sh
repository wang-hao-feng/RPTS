TEXT_PROMPT="cot_en6"
PROMPT_FN="image_path"
IMAGE_TOKEN_PROMPT="special_image_token"
MODEL="Qwen-VL-Chat"
OUTPUT="cot_en/${MODEL}-${TEXT_PROMPT}.json"

srun python main.py -ds ${RPTS_PATH} \
                    -tp ${TEXT_PROMPT} \
                    -pf ${PROMPT_FN} \
                    -itp ${IMAGE_TOKEN_PROMPT} \
                    -m ${MODEL} \
                    -o ${OUTPUT} \
                    -l "en" \
                    --bf16