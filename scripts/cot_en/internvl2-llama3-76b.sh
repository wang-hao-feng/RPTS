TEXT_PROMPT="cot_en4"
PROMPT_FN="normal"
IMAGE_TOKEN_PROMPT="special_image_token"
MODEL="InternVL2-Llama3-76B"
OUTPUT="cot_en/${MODEL}-${TEXT_PROMPT}.json"

srun python main.py -ds ${RPTS_PATH} \
                    -tp ${TEXT_PROMPT} \
                    -pf ${PROMPT_FN} \
                    -itp ${IMAGE_TOKEN_PROMPT} \
                    -m ${MODEL} \
                    -o ${OUTPUT} \
                    -l "en" \
                    --fp16 \
                    --load_in_8bit