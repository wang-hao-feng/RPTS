PROMPT1 = """\
I will give you some facts and a sentence. Please judge how well the sentence matches one of the given facts.

**Scoring Standard (0.0 to 1.0, in 0.1 increments):**
- **1.0**: Perfect match. The sentence is a direct paraphrase or exact description of a fact.
- **0.9**: Nearly perfect match. Only minor, negligible differences in wording or detail.
- **0.8**: Strong match. The sentence captures the core meaning of a fact with slight interpretation.
- **0.7**: Clear match. The sentence is consistent with a fact but may emphasize different aspects.
- **0.6**: Partial match. The sentence aligns with a fact in essence but has notable omissions or additions.
- **0.5**: Borderline match. The connection is weak or ambiguous; the sentence only loosely relates to a fact.
- **0.4**: Poor match. Significant discrepancy or only superficial similarity to any fact.
- **0.3**: Very poor match. The sentence contradicts or severely misrepresents the facts.
- **0.2-0.1**: Minimal to no match. Almost entirely unrelated or contradictory.
- **0.0**: Complete contradiction. The sentence states the opposite of a clear fact.


[FACTS]
[SENTENCE]

Please provide your rate, you only need to provide the score:
"""

PROMPT2 = """\
I will give you a reasoning, which consists of a context, some clues, and a conclusion. Please evaluate whether you can infer a conclusion based on the clues in the context and give a score for the correctness of this inference. 

**Scoring Standard (0.0 to 1.0, in 0.1 increments):**
- **1.0**: Flawless deduction. The conclusion follows necessarily and directly from the clues.
- **0.9**: Excellent inference. The conclusion is a near-certain, strongly justified implication.
- **0.8**: Good inference. The conclusion is highly reasonable with only a minor, acceptable leap.
- **0.7**: Reasonable inference. The conclusion is plausible, based on one clear, reasonable assumption.
- **0.6**: Acceptable inference. The conclusion is possible but relies on a noticeable logical leap.
- **0.5**: Borderline/Flawed inference. The connection is weak or tenuous. **(Threshold: Scores < 0.5 are considered incoherent.)**
- **0.4**: Poor inference. The conclusion is weakly supported and highly speculative.
- **0.3**: Very poor inference. The logical link is nearly broken or severely flawed.
- **0.2-0.1**: Extremely poor to no inference. The conclusion is arbitrary or unrelated.
- **0.0**: Contradiction. The conclusion is logically impossible given the clues.

[CONTEXT]
[CLUES]
[CONCLUSION]

Please provide your rate, you only need to provide the score:
"""