#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, List

from utils.common import PRED_CATEGORY_FIELD


# =========================================================
# 5) Prompt design
# =========================================================
_SCHEMA_ITEM = {
    "finest_category": "<required; category text only>",
    "attributes": {
        "<attribute_name>": ["<value_1>", "<value_2_if_any>"],
        "number": ["<integer string like 1, 2, 3, ... OR 'uncertain'>"]
    },
}

_CATEGORY_TEMPLATE = """
- Electronics: use [Brand]+[Model]+[Device] only when all included parts are explicit and identity-bearing
- Food/Snacks: prefer the shortest specific grounded item name; include brand/flavor/type only when explicit and necessary
- Cosmetics/Daily products: include brand/series/type/key spec only when explicit and identity-bearing
- Animals: specific species/breed only
- Other: use the most specific item name only, no descriptive features in the category
"""

_EXTRACT_TASK_HEADER = (
    "You are an expert extraction assistant. "
    "Extract all in-scope primary physical objects from the text into one JSON array.\n\n"
)

_DEFAULT_EXTRACT_SCOPE = "Extract all primary physical objects described in the text."

_EXTRACT_RULES_PROMPT = (
    "Rules:\n"
    "- Exclude generic background/environment/surface/light.\n"
    "- Do not extract low-information support/context objects as standalone objects unless the text itself focuses on them.\n"
    "- Minor accessories / holders / connectors / supports should usually stay in attributes or be omitted unless they are main described objects.\n"
    "- Support/context details may be kept in attributes when they help describe a primary object.\n"
    "- If an object has a labeled part/component/accessory, do not use the part/component/accessory name as the category of the whole object.\n"
    "- Ignore abnormal long repeated gibberish/OCR-noise sequences; never extract them as objects or use them as finest_category.\n"
    "- Output exactly one JSON array, no extra text.\n"
    "- Each object must contain:\n"
    f'  - "{PRED_CATEGORY_FIELD}"\n'
    '  - "attributes"\n'
    "- Put quantity/position inside attributes, not as top-level fields.\n"
    "- If quantity is stated or reasonably implied for a countable object, put it in attributes.number.\n"
    "- Use the exact integer string for specific quantity; use '1' when the text refers to one countable object instance; use '2' for 'a pair'.\n"
    "- For inherently paired objects, treat one full pair/set as quantity 2 when the text clearly refers to the complete pair, for example earbuds, earphones, shoes, socks, gloves, or chopsticks.\n"
    "- If the text says 'a set of earbuds' or another clearly paired object set, use '2' rather than '1'.\n"
    "- Use 'uncertain' for ranged, approximate, plural-but-unspecified, or genuinely unclear quantity, including sets/collections whose item count is not clear.\n"
    "- If position/order is explicitly stated, include it in attributes.position.\n"
    "\nCollection / duplicate handling:\n"
    "- Do not double count the same physical objects at both group and member levels.\n"
    "- When a counted group is partially described by more specific member objects, extract the specific members separately and set the group-level number to the remaining unspecified count.\n"
    "- Example: '12 smartphones, including one Mate 60 Pro and one Mate 50 Pro' -> smartphone number=10, Mate 60 Pro smartphone number=1, Mate 50 Pro smartphone number=1.\n\n"
    "- Do not merge objects with different explicit labels/series/specs/capacity/form-factor.\n\n"
    "Uncertainty:\n"
    "- If the text gives a certain category and then an uncertain subtype/model, keep the certain category only.\n"
    "- If the text says 'possibly X or Y', use the most specific common super-category supported by both X and Y.\n"
    "- If all category wording is uncertain, infer the most specific category that the text still supports.\n\n"
    f"{PRED_CATEGORY_FIELD} rules:\n"
    "- Category only; do not include generic appearance features.\n"
    "- Use the most specific category explicitly supported by the text.\n"
    "- Every word in category must be grounded in the text.\n"
    "- For finest_category, use an English category name only.\n"
    "- Do not copy non-English OCR text, labels, or packaging text into finest_category.\n"
    "- Prefer shorter grounded category names; do not add Brand/Flavor/Type words just to fit a template.\n"
    "- Keep visual/descriptive modifiers out of finest_category unless they are part of a lexicalized object name or an explicit identity-bearing product variant.\n"
    "- Put observed color, material, finish, texture, pattern, size, shape, position, orientation, state/condition, count of parts/components, and other visual details in attributes instead of finest_category.\n"
    "- Product brand/model/series/version may stay in finest_category only when explicit and identity-bearing; ordinary observed color or finish must still be attributes.\n"
    "- Example: 'orange matte iPhone 15 Pro Max with three camera lenses' -> finest_category='iPhone 15 Pro Max smartphone', attributes.color=['orange'], attributes.finish=['matte'], attributes.camera_lenses=['3'].\n"
    "- Example: 'white dog on the left' -> finest_category='dog', attributes.color=['white'], attributes.position=['left'].\n\n"
    "Guidance:\n"
    + _CATEGORY_TEMPLATE.strip()
    + "\n\nOutput schema:\n"
    + json.dumps(_SCHEMA_ITEM, ensure_ascii=True)
)

_SUB_PROMPTS: Dict[str, str] = {
    "animal": """
Extract only animals and plants described in the text.
""",
    "plant": """
Extract only animals and plants described in the text.
""",
    "electronic": """
Extract only electronic devices/products described in the text.
""",
    "car": """
Extract only cars or car-like vehicles described in the text.
""",
    "landmark": """
Extract only buildings/architecture. For architecture, keep the architecture itself as the category; do not extract attached parts such as clock face, window, door, roof, or tower details as separate objects.
""",
    "daily": """
Extract only daily-use consumer goods, such as snacks, drinks, cosmetics, toiletries, detergents, cleaning products, personal-care items, medicines/supplements, stationery, kitchenware, tableware, and household supplies.
""",
    "games": """
Extract only characters, creatures, weapons, and items in text; exclude background scenery and UI elements.
""",
}


def get_extraction_system_prompt(type_: str) -> str:
    sub = _SUB_PROMPTS.get(type_.strip().lower(), "").strip()
    scope = sub or _DEFAULT_EXTRACT_SCOPE
    return _EXTRACT_TASK_HEADER + "Scope:\n" + scope + "\n\n" + _EXTRACT_RULES_PROMPT


def build_extraction_messages(context: str, type_: str = "") -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": get_extraction_system_prompt(type_)},
        {
            "role": "user",
            "content": (
                context.strip()
                + "\n\nReturn ONLY one valid JSON array. No explanation. No thinking. The response must start with [ and end with ]. The final character must be ]."
            ),
        },
    ]


# =========================================================
# Candidate pairwise level-match prompt
# =========================================================
_CANDIDATE_PAIRWISE_LEVEL_MATCH_SYSTEM = """You are determining the relationship between a prediction and a target category.

Task: Analyze whether "Prediction" could refer to "Target" given the category hierarchy.

## Important Notes on Target Format:

The Target may contain parenthetical annotations like "fragrance (bottle)" or "Eye Cream (jar)".
- The parenthetical content (e.g., "bottle", "jar") is CONTEXTUAL INFORMATION, not a strict requirement
- Focus on the main category (e.g., "fragrance") and use parenthetical as hints
- "fragrance" and "perfume" are synonyms — treat them as equivalent

## Three-Way Decision:

**is_a** — Prediction IS the target, or is-a subtype/instance of target:
- "iPhone 15 Pro" is_a "iPhone" ✓ (subtype)
- "iPhone 15 Pro" is_a "smartphone" ✓ (broader category)
- "Samsung Galaxy S24" is_a "Samsung phone" ✓ (brand + type)
- "perfume" is_a "fragrance (bottle)" ✓ (synonym + parenthetical is hint)
- "fragrance" is_a "perfume" ✓ (synonyms)
- "perfume" is_a "Jo Malone Cologne" ✗ → should be "may_refer_to" (generic vs specific product)

**cannot_refer_to** — Prediction CANNOT refer to this target:
- Different domain entirely: "apple (fruit)" cannot_refer_to "Apple iPhone" ✗
- Different brand/family: "iPhone 15 Pro" cannot_refer_to "Samsung Galaxy" ✗
- Different generation/series: "iPhone 15" cannot_refer_to "iPhone 16 Pro" ✗
- Wrong type: "Android phone" cannot_refer_to "iPhone" ✗
- Different species: "cat" cannot_refer_to "dog" ✗
- "perfume" cannot_refer_to "Eye Cream (jar)" ✗ (cosmetics but different function)

**may_refer_to** — Prediction MIGHT refer to target, but uncertain:
- Prediction is more generic but could include target:
  - "smartphone" may_refer_to "iPhone" (smartphone could be iPhone)
  - "beverage" may_refer_to "Coca Cola" (could be, but uncertain)
- Uncertain if prediction is specific enough:
  - "red rose" may_refer_to "yellow rose" (color differs, might still match)
  - "electronics" may_refer_to "iPhone" (too generic, could match)
- Semantic overlap without strict is-a:
  - "perfume bottle" may_refer_to "bottle" (a type of)
  - "rose" may_refer_to "flower" (part of hierarchy)

## Decision Priority:
1. If you can CONFIDENTLY say "cannot_refer_to", return that
2. If Prediction is the same as or clearly a subtype of Target (including synonyms), return "is_a"
3. If uncertain (too generic, semantic overlap, could go either way), return "may_refer_to"

Return JSON with decision and reason."""



_CANDIDATE_PAIRWISE_DOMAIN_APPENDICES: Dict[str, str] = {
    "electric": """Identity examples for electronics:
- is_a: iPhone 15 Pro is_a iPhone, is_a smartphone
- may_refer_to: smartphone may_refer_to iPhone 15 Pro, not is_a (generic vs specific model)
- cannot_refer_to: iPhone 15 Pro cannot_refer_to Samsung Galaxy (different brand)
- cannot_refer_to: iPhone 15 cannot_refer_to iPhone 16 Pro (different generation/model)
- is_a: Samsung Galaxy is_a smartphone""",
    "daily": """Identity examples for daily products/snacks/drinks/cosmetics:
- is_a: Coca Cola Zero is_a Coca Cola, is_a beverage
- cannot_refer_to: Coca Cola Zero cannot_refer_to Pepsi (different brand)
- is_a: Strawberry Oreo is_a Oreo, is_a snack
- is_a: Dior Addict Lip Glow is_a lip product, is_a lip balm
- may_refer_to: lip product may_refer_to Dior Addict Lip Glow, not is_a (generic vs specific product)
- cannot_refer_to: YSL lipstick cannot_refer_to Dior Addict Lip Glow (different brand/product line)""",
    "car": """Identity examples for cars:
- is_a: Tesla Model 3 is_a Tesla, is_a electric car
- may_refer_to: car may_refer_to Tesla Model 3, not is_a (generic vs specific model)
- cannot_refer_to: Model 3 cannot_refer_to Model Y (different model)
- is_a: BMW 3 Series is_a BMW""",
    "animal": """Identity examples for animals:
- is_a: Golden Retriever is_a Dog, is_a Mammal
- may_refer_to: Dog may_refer_to Golden Retriever, not is_a (generic vs specific breed)
- cannot_refer_to: Husky cannot_refer_to Samoyed (different breed)""",
    "plant": """Identity examples for plants:
- is_a: Red rose is_a rose, is_a flower
- may_refer_to: flower may_refer_to red rose, not is_a (generic vs specific plant)
- cannot_refer_to: rose cannot_refer_to sunflower (different plant type)""",
    "building": """Identity examples for buildings:
- is_a: Church is_a Religious Building
- may_refer_to: building may_refer_to church, not is_a (generic vs specific building type)
- cannot_refer_to: Church cannot_refer_to Castle (different type)""",
    "game": """Identity examples for game entities:
- is_a: Hu Tao is_a Genshin Character
- may_refer_to: Genshin Character may_refer_to Hu Tao, not is_a (generic vs specific character)
- cannot_refer_to: Hu Tao cannot_refer_to Xiangling (different character)""",
}


def _candidate_pairwise_domain_appendix(type_: str) -> str:
    key = str(type_ or "").strip().lower()
    aliases = {
        "electronic": "electric",
        "electronics": "electric",
        "vehicle": "car",
        "landmark": "building",
        "games": "game",
    }
    key = aliases.get(key, key)
    return _CANDIDATE_PAIRWISE_DOMAIN_APPENDICES.get(key, "")


def build_candidate_pairwise_level_match_messages(
    pred_category: str,
    target_label: str,
    type_: str = "",
) -> List[Dict[str, str]]:
    """Build messages for three-way level matching."""
    system = _CANDIDATE_PAIRWISE_LEVEL_MATCH_SYSTEM + "\n\n" + _candidate_pairwise_domain_appendix(type_)

    lines = [
        f"Prediction: {pred_category!r}",
        "",
        f"Target: {target_label!r}",
        "",
        "Determine the relationship. Return JSON with decision (is_a/cannot_refer_to/may_refer_to) and reason.",
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]



_PAIR_ATTRIBUTE_SCORER_SYSTEM = """You are a pair-level attribute truthfulness and recall judge.

You will receive one predicted object's non-quantity attribute facts and several
category-legal GT candidates' non-quantity attribute facts. Category legality has
already been decided upstream. Do NOT use category labels, category depth, global
assignment, other predictions, or quantity allocation. Judge attributes only.

Task for each candidate GT:
1. Identify pred attribute facts that are explicitly described by the pred and are
   clearly consistent with / supported by the GT attributes.
2. Identify pred attribute facts that are explicitly described by the pred and are
   clearly inconsistent with / contradicted by the GT attributes.
3. Identify GT attribute facts that are explicitly recalled / covered by the pred
   attributes.

Rules:
- Be literal and conservative.
- Do not infer hidden facts.
- If a pred attribute is not clearly correct or clearly wrong from GT attributes,
  leave it out of both correct_pred_fact_indices and wrong_pred_fact_indices; it
  will be treated as uncertain / unverifiable.
- If a GT attribute is not clearly covered by pred attributes, leave it out of
  recalled_gt_fact_indices; it will be treated as unrecalled.
- The same pred fact index should not appear in both correct_pred_fact_indices
  and wrong_pred_fact_indices.
- Return one result for every input candidate_id. Do not omit candidates; use
  empty arrays when no facts match.
- Keep each reason under 20 words; no internal reasoning or self-correction.
- Output ONLY one JSON object with this structure:

{
  "candidates": {
    "g3": {
      "correct_pred_fact_indices": [0, 2],
      "wrong_pred_fact_indices": [1],
      "recalled_gt_fact_indices": [0, 2],
      "reason": "brief explanation"
    },
    "g5": {
      "correct_pred_fact_indices": [],
      "wrong_pred_fact_indices": [0],
      "recalled_gt_fact_indices": [],
      "reason": "brief explanation"
    }
  }
}
"""




def build_pair_attribute_scoring_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    candidate_ids = [str(c.get("candidate_id")) for c in payload.get("candidates", [])]
    header = (
        "Required candidate_ids: " + ", ".join(candidate_ids) + "\n"
        "Return exactly these candidate_ids under \"candidates\".\n\n"
    )
    return [
        {"role": "system", "content": _PAIR_ATTRIBUTE_SCORER_SYSTEM},
        {"role": "user", "content": header + json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
