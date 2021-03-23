import ast
import re
from math import pi

import spacy
# from Levenshtein import jaro
from more_itertools import locate, flatten
from nltk import Tree

# data_dir = '../../../../data'

nlp = spacy.blank("en")
operator_re = re.compile("<_ast.(\w+) object at ")
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
rx = re.compile(numeric_const_pattern, re.VERBOSE)


class AstParser(ast.NodeVisitor):

    def __init__(self):
        self.tree = None
        self.position = ()

    def continu(self, stmt):
        """Helper: parse a node's children"""
        super(AstParser, self).generic_visit(stmt)

    def parse(self, code):
        """Parse text into a tree and walk the result"""
        tree = ast.parse(code)
        for o in ast.iter_child_nodes(tree):
            if isinstance(o, ast.Expr):
                if isinstance(o.value, ast.BinOp):
                    self.tree = self.build_tree(o.value)
                elif isinstance(o.value, ast.Call):
                    self.tree = self.build_tree(o.value)
        return self.tree

    def build_tree(self, stmt_binop):
        sub_tree = Tree('ROOT', [])

        if isinstance(stmt_binop, ast.BinOp):
            left = self.build_tree(stmt_binop.left)

            right = self.build_tree(stmt_binop.right)

            if left:
                sub_tree.insert(0, left)

            if right:
                sub_tree.insert(1, right)

            sub_tree.set_label(operator_re.findall(str(stmt_binop.op))[0])
        elif isinstance(stmt_binop, ast.UnaryOp):
            sub_tree.insert(0, self.build_tree(stmt_binop.operand))
            sub_tree.set_label(operator_re.findall(str(stmt_binop.op))[0])
        elif isinstance(stmt_binop, ast.Call):
            sub_tree = Tree(stmt_binop.func.id, [self.build_tree(arg) for arg in stmt_binop.args])
        elif isinstance(stmt_binop, ast.Name):
            sub_tree = stmt_binop.id
        elif isinstance(stmt_binop, (ast.Constant, ast.Num)):
            sub_tree = stmt_binop.n
        elif isinstance(stmt_binop, ast.Attribute):
            sub_tree = stmt_binop.attr
        else:
            print(type(stmt_binop))

        return sub_tree


def percentage(code):
    return re.sub('([\d\.]+)\%', r'(\1/100)', code)


def fraction(code):
    return re.sub('([\d\.]+)\/([\d\.]+)', r'(\1/100)', code)


def equation2tree(parser, equation, answer, draw=False):
    a_s = percentage(answer)
    a = eval(a_s)

    equation = equation.replace('[', '(').replace(']', ')').strip("x=").replace("^", "**").strip()
    equation = percentage(equation)

    tree = parser.parse(equation)
    v = eval(equation)
    if draw:
        tree.draw()

    code = ast.parse(equation, mode="eval")
    tree_v = eval(compile(code, filename="", mode="eval"))

    return tree, a, v, tree_v


def text2int(textnum, numwords={}, ordinal=False):
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        # scales = ["hundred", "thousand", "million", "billion", "trillion"]

        # numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        # for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first': 1, 'second': 2, 'third': 3, 'fifth': 5, 'eighth': 8, 'ninth': 9, 'twelfth': 12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if ordinal:
            if word in ordinal_words:
                scale, increment = (1, ordinal_words[word])
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)
        else:
            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    curstring = re.sub('twice', '2 times', curstring)
    return curstring


def tree_mask(tree, tokens_list):
    mask = []
    num = []
    label = tree.label()
    # for n in tree.leaves():
    for pos in tree.treepositions('leaves'):
        n = tree[pos]
        cl = list(locate(tokens_list, lambda x: x == n))
        if not cl:
            return
        # elif len(cl) > 1:
        #     idx = (pos[-1] + 1) % 2
        #     nearest = pos[:-1] + (idx,)
        #     print(n, pos, nearest, tree[nearest], cl, tree)
        #     mask.append()
        #     num.append(n)
        else:
            mask.append(cl)
            num.append(n)
    # if len(mask) == len(tree.leaves()):
    #     return tree, label, num, mask
    return tree, label, num, mask


def check_tree(tree):
    label = tree.label()
    if tree.height() == 3 and label in ['Add', 'Sub']:
        if tree[(0,)] == '1' and tree[(1,)].label() == 'Div':
            new_tree = tree[(1,)]
            new_tree.set_label('{}Div'.format(label))
            return new_tree


def subtree_iter(item, tokens_list, subtree=False):
    tree, ans, val, tree_v = equation2tree(item['equation'], item['ans'])

    tree_updated = True
    tms = []
    while tree_updated:
        tree_updated = False

        if subtree:
            for pos in tree.treepositions('postorder'):
                if isinstance(tree[pos], Tree):
                    tm = tree_mask(tree[pos], tokens_list)
                    if tm:
                        tms.append(tm)
                    else:
                        new_tree = check_tree(tree[pos])
                        if new_tree:
                            if not pos:
                                tree = new_tree
                            else:
                                tree[pos] = new_tree
                            tree_updated = True
                            tms = []
        else:
            tm = tree_mask(tree, tokens_list)
            if tm:
                tms.append(tm)
            else:
                new_tree = check_tree(tree)
                if new_tree:
                    tree = new_tree
                    tree_updated = True
                    tms = []
    for tm in tms:
        tree, label, num, mask = tm
        tree.pretty_print()
        eval_ans = eval_tree(tree.pformat())
        # print(ans, val, eval_ans)
        # assert abs(ans - eval_ans) < 1e-4
        if len(set(flatten(mask))) == len(num) and len(list(flatten(mask))) == len(num):
            yield tm, eval_ans


label2op = {
    'NOp': '',
    'Sub': '-',
    'Div': '/',
    'Pow': '**',
    'Mult': '*',
    'Add': '+',
    'USub': '-',
}

func2op = {
    'add': 'Add',
    'subtract': 'Sub',
    'multiply': 'Mult',
    'rectangle_area': 'Mult',
    'divide': 'Div',
    'speed': 'Div',
    'power': 'Pow',
    'negate': 'USub',
    'inverse': ('Div', 1, None),
    'square_area': ('Pow', None, 2),
    'sqrt': ('Pow', None, 1 / 2),
    'square_edge_by_area': ('Pow', None, 1 / 2),
    'cube_edge_by_volume': ('Pow', None, 1 / 2),
    'volume_cube': ('Pow', None, 3),
    'surface_cube': ('Mult', ('Pow', None, 2), 6),
    'square_perimeter': ('Mult', None, 4),
    'rectangle_perimeter': ('Mult', ('Add', None, None), 2),
    'stream_speed': ('Div', ('Add', None, None), 2),
    'triangle_area': ('Div', ('Mult', None, None), 2),
    'triangle_perimeter': ('Add', ('Add', None, None), None),
    'surface_sphere': ('Mult', ('Pow', None, 2), ('Mult', 4, 'const_pi')),
    'volume_sphere': ('Mult', ('Pow', None, 3), ('Mult', ('Div', 4, 3), "const_pi")),
    'rhombus_area': ('Div', ('Mult', None, None), 2),
    'quadrilateral_area': ('Div', ('Mult', None, ('Add', None, None)), 2),
    'volume_cylinder': ('Mult', ('Mult', ('Pow', None, 2), 'const_pi'), None),
    'circle_area': ('Mult', ('Pow', None, 2), "const_pi"),
    'volume_cone': ('Div', ('Mult', ('Mult', None, 'const_pi'), None), 3),
    'circumface': ('Mult', ('Mult', None, 2), "const_pi"),
    'diagonal': ("Pow", ("Add", ("Pow", None, 2), ("Pow", None, 2)), 1 / 2),
    'volume_rectangular_prism': ('Mult', ('Mult', None, None), None),
    'original_price_before_loss': ('Div', None, ('Sub', 1, ('Div', None, 100))),
    'original_price_before_gain': ('Div', None, ('Add', 1, ('Div', None, 100))),
    'p_after_gain': ('Mult', ('Add', 1, ('Div', None, 100)), None),
    'square_edge_by_perimeter': ('Div', None, 4),
    'negate_prob': ('Sub', 1, None),
}

change_args_order = {
    'original_price_before_loss': (1, 0),
    'original_price_before_gain': (1, 0),
}

constants = {
    "const_pi": pi,
    "const_5": 5,
    "const_2": 2,
    "const_2.0": 2,
    "const_1": 1,
    "const_3": 3,
    "const_3.0": 3,
    "const_4": 4,
    "const_4.0": 4,
    "const_6": 6,
    "const_12": 12,
    "const_10": 10,
    "const_100": 100,
    "const_100.0": 100,
    "const_1000": 1000,
    "const_26": 26,
    "const_52": 52,
    "const_60": 60,
    "const_60.0": 60,
    "const_360": 360,
    "const_3600": 3600,
    "const_1_6": 1.6,
    "const_0.6": 0.6,
    "const_0_6": 0.6,
    "const_0_2778": 0.2778,
    "const_0.3937": 0.3937,
    "const_0_3937": 0.3937,
    "const_2.54": 2.54,
    "const_0.4535": 0.4535,
    "const_2.2046": 2.2046,
    "const_3_6": 3.6,
    "const_deg_to_rad": pi / 180,
    "const_180": 180,
    "const_0.5": 0.5,
    "const_0.25": 0.25,
    "const_0_25": 0.25,
    "const_0_33": 0.33
}


def parse_answer(answer):
    candidates = []
    for item in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\./:]?\d*(?:[eE][-+]?\d+)?", answer):
        candidates.append(item)

    if not candidates:
        return 'none'

    if len(candidates) == 1:
        obj = candidates[0]
        if isinstance(obj, str) and obj.startswith('0') and not obj.startswith('0.'):
            obj = f"0.{obj}"
        return obj
    elif len(candidates) == 2:
        if '/' in answer or ':' in answer:
            return f'{candidates[0]} / {candidates[1]}'
        else:
            return candidates[0]
    else:
        return f'{candidates[0]} + {candidates[1]} / {candidates[2]}'


def operator_adaptation(ops):
    if not isinstance(ops, tuple):
        if ops in func2op:
            ops = func2op.get(ops, ops)
            if isinstance(ops, str):
                ops = (ops, None, None)
        else:
            return constants.get(ops, ops)

    op, *args = ops

    new_tree = Tree(op, [])

    for arg in args:
        new_tree.append(operator_adaptation(arg))
    return new_tree


def update_tree(tree):
    label = tree.label()
    new_tree = operator_adaptation(label)
    if not isinstance(new_tree, Tree):
        return new_tree

    if label in change_args_order:
        tree = [tree[i] for i in change_args_order[label]]

    for p in new_tree.treepositions():
        if not new_tree[p]:
            tmp = tree.pop(0)
            if isinstance(tmp, Tree):
                new_tree[p] = update_tree(tmp)
            else:
                new_tree[p] = constants.get(tmp, tmp)
    return new_tree


def pformat_flat(tree, nodesep="", parens="()", quotes=False):
    childstrs = []
    for child in tree:
        if isinstance(child, Tree):
            childstrs.append(pformat_flat(child, nodesep, parens, quotes))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            childstrs.append("%s" % child)
        else:
            childstrs.append(repr(child))
    if isinstance(tree._label, str):
        return "%s %s %s %s %s" % (
            parens[0],
            tree._label,
            nodesep,
            " ".join(childstrs),
            parens[1],
        )
    else:
        return "%s %s %s %s %s" % (
            parens[0],
            repr(tree._label),
            nodesep,
            " ".join(childstrs),
            parens[1],
        )


def eval_tree(tree, evaluation=False):
    tree_copy = Tree.fromstring(tree)
    tree_updated = True
    while tree_updated:
        tree_updated = False
        for pos in tree_copy.treepositions('postorder'):
            if isinstance(tree_copy[pos], Tree):
                leaves = tree_copy[pos].leaves()
                label = tree_copy[pos].label()
                if label == 'USub':
                    code = "- {}".format(leaves[0])
                else:
                    if label == 'Pow' and float(leaves[1]) > 3 and evaluation:
                        return -1
                    code = "{} {} {}".format(leaves[0], label2op[label], leaves[1])

                val = eval(code)
                tree_updated = True
                if not pos:
                    return val
                else:
                    tree_copy[pos] = val


if __name__ == '__main__':
    # a = "1-(-(1/2))"
    # t = parser.parse(a)
    # t.draw()
    # for b in iter_process('train'):
    #     print(b)

    for b in iter_process('train', False):
        print(b)
