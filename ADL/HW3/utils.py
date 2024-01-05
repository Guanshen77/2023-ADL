from transformers import BitsAndBytesConfig
import torch



def get_prompt_0(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"

def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。接下來要做翻譯任務，請根據USER指令進行文言文及白話文的轉換，其中文言文是在中國古代語言模式，白話文又稱為現代文是現代的語言模式。USER: {instruction} ASSISTANT:"

def get_prompt_few_shot(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。接下來要做翻譯任務，請根據USER指令進行文言文及白話文的轉換， 其中文言文是在中國古代語言模式，白話文又稱為現代文是現代的語言模式 \
    請學習以下翻譯例子進行轉換 \
    將“正月，甲子朔，鼕至，太後享通天宮；赦天下，改元”翻譯成現代文。 答案：聖曆元年正月，甲子朔，鼕至，太後在通天宮祭祀；大赦天下，更改年號。 \
    將“雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。”翻譯成文言文。 答案：雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。\
    將下麵句子翻譯成文言文：令、錄、簿、尉等職官有年老病重的人允許彈勃。 答案：令、錄、簿、尉諸職官有耄耋篤疾者舉劾之。 \
    翻譯成現代文：\n士匄請見，弗內。答案：士匄請求進見，荀偃不接見。 \
    翻譯成文言文：\n富貴貧賤都很尊重他。答案：貴賤並敬之。 \
    USER: {instruction} ASSISTANT:" 

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''

    quantization_config=BitsAndBytesConfig(
        load_in_4bit= 4,
        load_in_8bit= 4 == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    return quantization_config
