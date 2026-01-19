COT_ZH1 = """\
我将给你一段上下文、一些文本线索、一些图像线索和一段声明，请你告诉我，根据这些文本线索和图像线索，你所能推理出的结论是支持给定声明还是反对给定声明，你给出详细且严谨的推理过程，并回答支持或反对。
一个简单的推理如下：
```
结论0：根据图片0，________
结论1：根据结论0和文本0，______
```
请仿照上面的格式给出你的推理。在你给出的推理中，每步推理都需要给出推理的理由，理由需要说明这步推理是根据哪张图片、哪条文本线索、哪个结论得来。
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
现在请一步步给出推理过程，结论从0开始计数。
"""

COT_ZH2 = """\
我将给你一段上下文、一些文本线索、一些图像线索和一段声明，请你告诉我，根据这些文本线索和图像线索，你所能推理出的结论是支持给定声明还是反对给定声明，你给出详细且严谨的推理过程，并回答支持或反对。
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
在你给出的推理中，每步推理都需要给出推理的理由以及结论，理由需要复述一遍这步推理你所使用的线索以及结论。现在请一步步给出推理过程。
"""

COT_ZH3 = """\
让我们来玩一个推理游戏，我将给你一段上下文、一些文本线索、一些图像线索和一段声明，请你告诉我，根据这些文本线索和图像线索，你所能推理出的结论是支持给定声明还是反对给定声明，你需要给出详细且严谨的推理过程。
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
在你给出的推理中，每步推理都需要给出推理的理由以及结论,理由需要说明这步推理是根据哪张图片、哪条文本线索、哪个结论得来，现在请一步步给出推理过程:
"""

COT_ZH4 = """\
让我们来玩一个推理游戏。下面这段文本中，上下文描述这个推理游戏的背景以及图像的标题；文本线索和图像是当前能够确定的事实。
[CONTEXT]
[CLUE]
[IMAGE]
首先，你需要尽可能详细的描述图像，然后请你根据图像中的细节和文本线索，推理“[STATEMENT]”是否正确。
你需要给出详细且严谨的推理过程，每步推理都需要给出推理的理由以及结论，理由需要说明这步推理是根据哪张图片、哪条文本线索、哪个结论得来，图片、文本线索、结论都从0开始计数。
"""

COT_ZH5 = """\
[IMAGE]
让我们来玩一个推理游戏。下面这段文本中，上下文描述这个推理游戏的背景以及图像的标题；文本线索和图像是当前能够确定的事实。
[CONTEXT]
[CLUE]
首先，你需要尽可能详细的描述图像，然后请你根据图像中的细节和文本线索，推理“[STATEMENT]”是否正确。
你需要给出详细且严谨的推理过程，每步推理都需要给出推理的理由以及结论。
"""

COT_EN1 = """\
I will give you a context, some textual clues, some visual clues, and a statement. Please tell me whether the conclusion you can infer based on these textual and visual clues agree or disagree the given statement. Please provide a detailed and rigorous reasoning process and answer agree or disagree.
A simple inference is as follows:
```
Conclusion 0: According to image 0________
Conclusion 1: Based on Conclusion 0 and Text 0______
```
Please provide your reasoning following the format above. In the reasoning you provide, each step requires a reason for the reasoning, which needs to explain which image, which textual clue, and which conclusion the reasoning is based on.
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
Let's think step by step, and each step of reasoning requires giving reasons for reasoning.
"""

COT_EN2 = """\
I will give you a context, some textual clues, some visual clues, and a statement. Please tell me whether the conclusion you can infer based on these textual and visual clues agree or disagree the given statement. Please provide a detailed and rigorous reasoning process and answer agree or disagree.
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
In the reasoning you provide, you need to give reasons and conclusions for each step of the reasoning, and the reasons need to be repeated along with the clues and conclusions you used in this step of reasoning. Now please provide the reasoning process step by step.
"""

COT_EN3 = """\
Let's play a reasoning game. I will give you a context, some textual clues, some visual clues, and a statement. Please tell me whether the conclusion you can infer based on these textual and visual clues agree or disagree the given statement. You need to provide a detailed and rigorous reasoning process.
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
In the reasoning you provided, each step requires a reason and conclusion for the reasoning. The reason needs to explain which image, which textual clue, and which conclusion the reasoning was based on. Now, please provide the reasoning process step by step:
"""

COT_EN4 = """\
Let's play a reasoning game. In the following text, the context describes the background of this reasoning game and the title of the image; Textual clues and visual are currently determinable facts.
[CONTEXT]
[CLUE]
[IMAGE]
Firstly, you need to provide a detailed description of the image as much as possible, and then infer whether "[STATEMENT]" is correct based on the details in the image and textual clues.
You need to provide a detailed and rigorous reasoning process, with reasons and conclusions for each step of reasoning. The reasons should explain which image, textual clue, and conclusion this step of reasoning is based on, and the counting of images, text clues, and conclusions starts from 0.
"""

COT_EN5 = """\
[IMAGE]
Let's play a reasoning game. In the following text, the context describes the background of this reasoning game and the title of the image; Text clues and images are currently determinable facts.
[CONTEXT]
[CLUE]
Firstly, you need to provide a detailed description of the image as much as possible, and then infer whether "[STATEMENT]" is correct based on the details and textual clues in the image.
You need to provide a detailed and rigorous reasoning process, with reasons and conclusions for each step of reasoning.
"""

COT_EN6 = """\
I will give you a context, some textual clues, some visual clues, and a statement. Please tell me whether the conclusion you can infer based on these textual and visual clues agree or disagree the given statement. Please provide a detailed and rigorous reasoning process and answer agree or disagree.
A simple inference is as follows:
```
Conclusion 0: According to image 0, we can infer that ...
Conclusion 1: Based on Conclusion 0 and Text 0, we can infer that ...
Conclusion 2: Based on Text 1 and Text 2, we can infer that ...
Conclusion 3: Based on Conclusion 1 and Conclusion 2 wen can infer that these textual and visual clues agree/disagree the given statement.
```
Please provide your reasoning following the format above. In the reasoning you provide, each step requires a reason for the reasoning, which needs to explain which image, which textual clue, and which conclusion the reasoning is based on.
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
Please provide your reasoning following the format above.
"""

COT_EN7 = """\
I will give you a context, some textual clues, some visual clues, and a statement. Please tell me whether the conclusion you can infer based on these textual and visual clues agree or disagree the given statement. Please provide a detailed and rigorous reasoning process and answer agree or disagree.
[CONTEXT]
[CLUE]
[IMAGE]
[STATEMENT]
In the reasoning you provide, you need to give reasons and conclusions for each step of the reasoning, and the reasons need to be repeated along with the clues and conclusions you used in this step of reasoning.
"""


prompt_map = {
    'cot_zh1': COT_ZH1, 
    'cot_zh2': COT_ZH2, 
    'cot_zh3': COT_ZH3, 
    'cot_zh4': COT_ZH4, 
    'cot_zh5': COT_ZH5, 
    'cot_en1': COT_EN1, 
    'cot_en2': COT_EN2, 
    'cot_en3': COT_EN3, 
    'cot_en4': COT_EN4, 
    'cot_en5': COT_EN5,
    'cot_en6': COT_EN6,
    'cot_en7': COT_EN7,
}