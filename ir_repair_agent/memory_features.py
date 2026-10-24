"""Retrieval features for repair memory."""

from __future__ import annotations

import re


def memory_error_pattern(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    matches = re.findall(r"(?:Error|Exception):?\s*(.+)$", text, re.M)
    if matches:
        return matches[-1].strip()[:500]
    diffs = re.findall(r"\[(MISSING|EXTRA|MISMATCH)\]\s*(.*)$", text, re.M)
    if diffs:
        return " | ".join(
            f"{status}: {reason.strip()}" for status, reason in diffs
        )[:500]
    return text[:500]


def structural_tokens(pattern: str) -> set[str]:
    frags = re.findall(r"(?:expected|implemented)\s*=\s*([^|]+)", pattern or "")
    if not frags:
        frags = [pattern or ""]
    tokens: set[str] = set()
    for frag in frags:
        text = (frag or "").lower()
        if "<=" in text:
            tokens.add("<=")
        elif ">=" in text:
            tokens.add(">=")
        elif "==" in text:
            tokens.add("==")
        elif "=" in text:
            tokens.add("=")
        elif "<" in text:
            tokens.add("<")
        elif ">" in text:
            tokens.add(">")
        if re.search(r"sum\s*\(|quicksum\s*\(", text):
            tokens.add("SUM")
        if re.search(r"\bt\s*-\s*1\b|\[\s*t\s*-\s*1\s*\]", text):
            tokens.add("RECUR")
        without_lag = re.sub(r"\[\s*t\s*-\s*1\s*\]|\bt\s*-\s*1\b", "", text)
        if re.search(r"\d+(\.\d+)?", without_lag):
            tokens.add("CONST")
        if "*" in without_lag:
            tokens.add("MUL")
    return tokens


def constraint_skeleton(fragment: str) -> str:
    text = (fragment or "").lower()
    text = re.sub(r"\[\s*t\s*-\s*1\s*\]", "\u00a7P\u00a7", text)
    text = re.sub(r"\[\s*t\s*\]", "\u00a7C\u00a7", text)
    text = re.sub(r"\b[a-z_][a-z0-9_]*\b", "x", text)
    text = re.sub(r"\d+(\.\d+)?", "C", text)
    text = text.replace("\u00a7P\u00a7", "[prev]").replace("\u00a7C\u00a7", "[cur]")
    return re.sub(r"\s+", "", text)


def structural_skeletons(pattern: str) -> set[str]:
    frags = re.findall(r"(?:expected|implemented)\s*=\s*([^|]+)", pattern or "")
    if not frags:
        frags = [pattern or ""]
    output: set[str] = set()
    for fragment in frags:
        fragment = (fragment or "").strip()
        if not fragment or fragment.lower() in {
            "n/a", "na", "none", "-", "null",
        }:
            continue
        output.add(constraint_skeleton(fragment))
    return output

