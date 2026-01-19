import re
from queue import Queue

def get_index(text:str):
    idx = re.findall(r'\d+', text)
    if len(idx) == 0:
        return -1
    return int(idx[0])

def is_image(antecedent:str):
    return '图' in antecedent or 'image' in antecedent.lower()

def is_text(antecedent:str):
    return ('文' in antecedent and '上下文' not in antecedent) or 'text' in antecedent.lower()

def is_conclusion(antecedent:str):
    return '结论' in antecedent or 'conclusion' in antecedent.lower()

def is_context(antecedent:str):
    return '上下文' in antecedent or 'context' in antecedent.lower()

class ReasoningNode:
    def __init__(self, 
                 score:float, 
                 text:str='', 
                 index:int=0, 
                 node_type:str='conclusion') -> None:
        self.score = score
        self.text = text
        self.children = []
        self.index = index
        self.node_type = node_type
    
    @property
    def is_leaf(self):
        return len(self.children) == 0

class ReasoningTree:
    def __init__(self, 
                 visual_clues_num:int, 
                 textual_clues:list[str], 
                 reasonings:list[str], 
                 scores:list[float]) -> None:
        assert len(scores) == len(reasonings)
        textual_clue_nodes = [ReasoningNode(1.0, clue, index=i, node_type='text') for i, clue in enumerate(textual_clues)]
        visual_clue_nodes = [ReasoningNode(1.0, index=i, node_type='image') for i in range(visual_clues_num)]
        reasoning_nodes = []
        context_node = ReasoningNode(1.0, index=-1, node_type='context')
        error_node = ReasoningNode(0.0, index=-1, node_type='error')
        def antecedent2node(antecedent:str):
            idx = get_index(antecedent)
            if is_image(antecedent):
                return visual_clue_nodes[idx] if idx < len(visual_clue_nodes) else error_node
            if is_text(antecedent):
                return textual_clue_nodes[idx] if idx < len(textual_clue_nodes) else error_node
            if is_conclusion(antecedent):
                return reasoning_nodes[idx] if idx < len(reasoning_nodes) else error_node
            if is_context(antecedent):
                return context_node
        for reasoning, score in zip(reasonings, scores):
            antecedent, consequent = reasoning.split('->')
            consequent_node = ReasoningNode(score, consequent, node_type='conclusion')
            consequent_node.children = [antecedent2node(a) for a in antecedent.split('+')]
            reasoning_nodes.append(consequent_node)
        self.root = reasoning_nodes[-1] if len(reasoning_nodes) > 0 else None
    
    @property
    def non_leaf_num(self) -> int:
        counter = 0
        queue = Queue()
        queue.put(self.root, block=False)
        while not queue.empty():
            node = queue.get()
            counter += int(not node.is_leaf)
            for child in node.children:
                queue.put(child)
        return counter