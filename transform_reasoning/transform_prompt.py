REASONING_TRANSFORM_PROMPT_ZH = """
我将给你一段语言模型的推理，请你将这段推理整理成给定的格式，格式要求如下：
每一步推理都以“前件i + 前件j + ... -> 结论k”的格式表示；
前件只能是“文本”、“图片”或者“结论”；
i, j, k都从0开始计数。
下面给出一个整理完成的例子：
```
图片0 -> 结论0：A车亮了红色的灯，B车在A车后方
结论0 + 文本0 + 文本3 -> 结论1：A车亮的是刹车灯，B车在A车后方
结论1 + 文本2 -> 结论2：起步后，两车会远离
结论2 + 文本1 -> 结论3：A车起步后不会撞到B车
```

请你按照要求整理下述推理，这个推理总共涉及[IMAGE_NUM]张图片：
[TEXTUAL_CLUE]
[RESULT]

推理中可能会重复输出一些内容，请你忽略掉重复的部分。推理中可能会复述文本线索，请你不要将复述的文本线索视为结论。
图片作为前件时，不能有其他前件，如遇到其他前件，你需要将推理拆成两部分。
你只需要输出整理完成后的文本，不要输出额外的内容。你必须保证整理完后的文本意思与整理之前完全相同。
"""

ANSWER_EXTRACT_PROMPT_ZH = """
我将给你一段文字和一句声明，请你告诉我这段文字是支持还是不支持给定声明，或是无法判断。
有时文字会直接告诉你它是否支持给定声明，你必须复述它的判断。
[STATEMENT]
[RESULT]
请你给出你的答案，你只需要回答“支持”，“反对”和“无法判断”中的一个：
"""

REASONING_TRANSFORM_PROMPT_EN = """
I will provide you with a language model inference. Please organize this inference into the given format, with the following format requirements:
Each step of reasoning is represented in the format of "Antecedent i + Antecedent j+... -> conclusion k";
The antecedent can only be "Text", "Image" or "Conclusion";
Both i, j and k start counting from 0.
Here is a completed example of organization:
```
Image 0 -> Conclusion 0: Car A has a red light on, and car B is behind car A
Conclusion 0 + Text 0 + Text 3 -> Conclusion 1: The brake light on car A is on, and car B is behind car A
Conclusion 1 + Text 2 -> Conclusion 2: After starting, the two cars will move away
Conclusion 2 + Text 1 -> Conclusion 3: Car A will not collide with car B after starting
```

Please organize the following reasoning as required, which involves a total of [IMAGE NUM] images:
[TEXTUAL_CLUE]
[RESULT]
During reasoning, there may be some repetitive outputs, please ignore the duplicated parts. 
During reasoning, textual clues may be repeated. Please do not consider the repeated textual clues as conclusions.
When using images as antecedent, there should be no other antecedent. If encountering other antecedent, you need to split the reasoning into two parts.
You only need to output the organized text, do not output any additional content. You must ensure that the meaning of the organized text is exactly the same as before.
"""

ANSWER_EXTRACT_PROMPT_EN = """
I will provide you with a sentence and a statement. Please tell me whether the sentence agree the given statement, disagree it, or cannot be determined.
Some times the sentence will directly tell you he is agree/support or disagree/(not support) with the statement, you must repeat his judgement as answer.
[STATEMENT]
[RESULT]
You only need to answer "agree", "disagree" or "cannot be determined":
"""