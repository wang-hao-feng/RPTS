TASK=cot_zh
TEXT_PROMPT="${TASK}1"
PROMPT_FN="normal"
IMAGE_TOKEN_PROMPT="without"
MODEL="GPT4o"
OUTPUT="${TASK}/${MODEL}-${TEXT_PROMPT}.json"

python main.py -ds ${BATCH_SIZE} \
               -tp ${TEXT_PROMPT} \
               -pf ${PROMPT_FN} \
               -itp ${IMAGE_TOKEN_PROMPT} \
               -m ${MODEL} \
               -o ${OUTPUT} \
               -l "zh"