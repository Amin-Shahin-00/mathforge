"""Parse model-written maths without ever executing model-written code.

SymPy's ``sympify``/``parse_expr`` call Python ``eval``. Text produced by a
neural network must never reach that unchecked, so we (1) whitelist the
characters and identifiers, (2) evaluate with an empty ``__builtins__`` and a
dictionary containing only maths functions, and (3) cap the length.
"""

from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

ALLOWED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "arcsin": sp.asin, "arccos": sp.acos,
    "arctan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "asinh": sp.asinh,
    "acosh": sp.acosh, "atanh": sp.atanh, "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs, "factorial": sp.factorial,
    "binomial": sp.binomial, "Rational": sp.Rational, "erf": sp.erf,
}
ALLOWED_CONSTS = {"pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I, "oo": sp.oo}
_SAFE_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/^().,\[\]\s]*$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


class UnsafeExpression(ValueError):
    pass


def safe_parse(text: str, variables: tuple[str, ...] = ("x",), max_len: int = 400):
    text = text.strip().rstrip(".").strip()
    if text.startswith("$") and text.endswith("$"):
        text = text.strip("$")
    if len(text) == 0 or len(text) > max_len:
        raise UnsafeExpression("empty or too long")
    if not _SAFE_CHARS.match(text) or "__" in text:
        raise UnsafeExpression("illegal characters")
    for name in _IDENT.findall(text):
        if name not in ALLOWED_FUNCS and name not in ALLOWED_CONSTS and name not in variables:
            # allow implicit multiplication like "xy" only for declared variable letters
            if not all(ch in variables for ch in name):
                raise UnsafeExpression(f"unknown identifier {name!r}")
    local = {v: sp.Symbol(v) for v in variables}
    local.update(ALLOWED_FUNCS)
    local.update(ALLOWED_CONSTS)
    glob = {"__builtins__": {}, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
            "Symbol": sp.Symbol, "Function": sp.Function, "Tuple": sp.Tuple}
    return parse_expr(text, local_dict=local, global_dict=glob, transformations=TRANSFORMS,
                      evaluate=True)


# (regex, literal replacement), applied in order
_LATEX_WORDS = [
    (r"\\[dt]frac", "\\frac"), (r"\\left", ""), (r"\\right", ""), (r"\\cdot", "*"),
    (r"\\times", "*"), (r"\\div", "/"), (r"\\ln", "log"), (r"\\log", "log"), (r"\\exp", "exp"),
    (r"\\arcsin", "asin"), (r"\\arccos", "acos"), (r"\\arctan", "atan"), (r"\\sinh", "sinh"),
    (r"\\cosh", "cosh"), (r"\\tanh", "tanh"), (r"\\sin", "sin"), (r"\\cos", "cos"),
    (r"\\tan", "tan"), (r"\\cot", "cot"), (r"\\sec", "sec"), (r"\\csc", "csc"), (r"\\pi", "pi"),
    (r"\\infty", "oo"), (r"\\displaystyle", ""), (r"\\quad", ""), (r"\\[,;! ]", ""),
]


def latex_to_text(s: str) -> str:
    """Convert the LaTeX that maths models like to emit into SymPy-style text.

    Only rewrites notation; the result still has to pass ``safe_parse``.
    """
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(?:text|mathrm|operatorname)\{\s*([A-Za-z]+)\s*\}",
               lambda m: "E" if m.group(1) == "e" else m.group(1), s)
    for pat, rep in _LATEX_WORDS:
        s = re.sub(pat + (r"(?![A-Za-z])" if pat[-1].isalpha() else ""), lambda m, r=rep: r, s)
    s = s.replace("\\{", "{").replace("\\}", "}")
    prev = None
    while prev != s:                       # innermost first, so nesting works
        prev = s
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
        s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9])", r"sqrt(\1)", s)
    s = re.sub(r"\|([^|]*)\|", r"(Abs(\1))", s)
    # "sin x" -> "sin(x)": bind a bare argument to its function (LaTeX habit)
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|asin|acos|atan|log|exp|sqrt)\s+"
               r"([A-Za-z0-9]+(?:\*\*[A-Za-z0-9]+)?)", r"\1(\2)", s)
    s = s.replace("{", "(").replace("}", ")")
    return s


def normalize_answer(ans: str, kind: str) -> str:
    """Tidy a final answer: LaTeX, 'F(x) = ...', '+ C', 'x = 1 or x = 2', sets."""
    if "\\" in ans or "{" in ans:
        ans = latex_to_text(ans)
    ans = ans.strip().rstrip(".").strip()
    if kind in ("integrate", "sum"):
        if "=" in ans:
            ans = ans.split("=")[-1]
        ans = re.sub(r"\s*\+\s*C\s*$", "", ans.strip())
    elif kind == "solve":
        ans = re.sub(r"\b(or|and)\b", ",", ans)
        ans = re.sub(r"\bx\s*=\s*", "", ans)
    return ans.strip()


def extract_answer(completion: str, tag: str = "ANSWER:") -> str | None:
    """Last ``ANSWER: ...`` line, or the last \\boxed{...} as a fallback."""
    lines = [l for l in completion.strip().splitlines() if tag in l]
    if lines:
        return lines[-1].split(tag, 1)[1].strip()
    start = completion.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(completion)):      # proper brace matching, any nesting depth
        if completion[j] == "{":
            depth += 1
        elif completion[j] == "}":
            depth -= 1
            if depth == 0:
                return completion[i:j]
    return None
