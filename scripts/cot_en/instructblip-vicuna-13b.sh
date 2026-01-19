TEXT_PROMPT="cot_en7"
PROMPT_FN="normal"
IMAGE_TOKEN_PROMPT="without"
MODEL="InstructBLIP-vicuna-13b"
OUTPUT="cot_en/${MODEL}-${TEXT_PROMPT}.json"

srun python main.py -ds ${RPTS_PATH} \
                    -tp ${TEXT_PROMPT} \
                    -pf ${PROMPT_FN} \
                    -itp ${IMAGE_TOKEN_PROMPT} \
                    -m ${MODEL} \
                    -o ${OUTPUT} \
                    -l "en" \
                    --fp16