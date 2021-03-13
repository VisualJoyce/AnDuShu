import ast
import math
import re

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

    equation = equation.replace('[', '(').replace(']', ')').strip("x=").replace("^", "**")
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


# def iter_process(data_type, subtree):
#     error_count = []
#     with open(os.path.join(data_dir, 'math23k', 'Math23k', 'math23k_{}_en.json'.format(data_type)), 'r',
#               encoding='utf8') as f:
#         data = json.load(f)
#
#         for item in data:
#             if item['id'] in ['4303', '9718', '9761', '10431', '12495', '17520']:
#                 continue
#
#             print('{} --------------------------{} {}'.format(item['id'], item['equation'], item['ans']))
#
#             item['en'] = re.sub('([\d\.]+)times', r'\1 times', item['en'])
#             item['en'] = re.sub('([\d\.]+) %', r'\1%', item['en'])
#             # item['en'] = re.sub('([\d\.]+) \/ ([\d\.]+)', r'\1/\2', item['en'])
#             item['en'] = re.sub('(\d+),(\d+)', r'\1\2', item['en']).replace('%', ' / 100')
#             item['ans'] = re.sub('(\d+)\(\(', r'\1+((', item['ans'])
#             print(item)
#             tokens_list = ['--DELIMITER--'] + [t.text for t in nlp(text2int(item['en']))]
#             print(tokens_list)
#             # try:
#             for b in subtree_iter(item, tokens_list, subtree):
#                 yield b
#             # except TypeError as e:
#             #     error_count.append((str(e), item))
#         print(json.dumps(error_count, ensure_ascii=False, indent=2))
#         print(len(error_count))


#
# label2op = {'NOp': '', 'Sub': '-', 'Div': '/', 'Pow': '**', 'Mult': '*', 'Add': '+', 'USub': '-',
#             'SubDiv': '',
#             'AddDiv': ''}


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
}

constants = {
    "const_pi": math.pi,
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
    "const_deg_to_rad": math.pi / 180,
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


# def eval_tree(tree):
#     tree_copy = Tree.fromstring(tree)
#     tree_updated = True
#     while tree_updated:
#         tree_updated = False
#         for pos in tree_copy.treepositions('postorder'):
#             if isinstance(tree_copy[pos], Tree):
#                 leaves = [constants.get(lf, lf) for lf in tree_copy[pos].leaves()]
#                 label = tree_copy[pos].label()
#
#                 if label in ['USub', 'negate']:
#                     code = "- {}".format(leaves[0])
#                 elif label in ['sqrt', 'square_edge_by_area']:
#                     code = "({}) ** 0.5".format(leaves[0])
#                 elif label in ['cube_edge_by_volume']:
#                     code = "({}) ** (1 / 3)".format(leaves[0])
#                 elif label in ['square_area']:
#                     code = "({}) ** 2".format(leaves[0])
#                 elif label in ['square_perimeter']:
#                     code = "({}) * 4".format(leaves[0])
#                 elif label in ['surface_cube']:
#                     code = "6 * ({}) ** 2".format(leaves[0])
#                 elif label in ['volume_cube']:
#                     code = "({}) ** 3".format(leaves[0])
#                 elif label in ['inverse']:
#                     code = "1 / ({})".format(leaves[0])
#                 elif label in ['stream_speed']:
#                     code = "({} + {}) / 2".format(leaves[0], leaves[1])
#                 elif label in ['rectangle_perimeter']:
#                     code = "({} + {}) * 2".format(leaves[0], leaves[1])
#                 elif label in ['triangle_perimeter']:
#                     code = "({} + {} + {})".format(leaves[0], leaves[1], leaves[2])
#                 elif label in ['triangle_area']:
#                     code = "({} * {}) / 2".format(leaves[0], leaves[1])
#                 else:
#                     code = "{} {} {}".format(leaves[0], label2op[label], leaves[1])
#
#                 val = eval(code)
#
#                 tree_updated = True
#                 if not pos:
#                     return val
#                 else:
#                     tree_copy[pos] = val
def update_tree(tree):
    label = tree.label()
    ops = func2op.get(label, label)

    if isinstance(ops, tuple):
        if isinstance(ops[1], tuple):
            op, *args = ops

            if not args[1]:
                args_1 = tree.pop()
            else:
                args_1 = args[1]

            op2, *args2 = ops[1]
            tree.set_label(op2)
            for i, a in enumerate(args2):
                if a:
                    tree.insert(i, a)
            tree = Tree(op, [tree, args_1])
        else:
            op, *args = ops
            tree.set_label(op)
            for i, a in enumerate(args):
                if a:
                    tree.insert(i, a)
    else:
        tree.set_label(ops)

    for i, child in enumerate(tree):
        if isinstance(child, Tree):
            tree[i] = update_tree(child)
        else:
            tree[i] = constants.get(child, child)
    return tree


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
