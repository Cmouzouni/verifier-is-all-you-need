"""30 simple multi-approach tasks for Phase A experiments.

Each task has:
- A clear correct answer (externally verifiable)
- 4 named solution frameworks (approaches)
- Per-framework system prompts
- Difficulty: easy enough for 3B models
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    problem: str
    ground_truth: str
    domain: str
    difficulty: str  # easy | medium
    frameworks: list[str]
    framework_prompts: dict[str, str] = field(default_factory=dict)

    def check(self, answer: str) -> bool:
        """Check if answer matches ground truth (normalized)."""
        norm = lambda s: s.strip().lower().replace(",", "").replace("$", "").replace(" ", "")
        gt = norm(self.ground_truth)
        ans = norm(answer)
        # Exact match
        if gt == ans:
            return True
        # Numeric comparison
        try:
            return abs(float(gt) - float(ans)) < 1e-3
        except (ValueError, TypeError):
            pass
        # Check if ground truth appears in answer
        return gt in ans


# ── Framework prompt templates ─────────────────────────────────────────
# Each framework gets a specific system prompt that constrains the approach

MATH_FRAMEWORKS = ["algebraic", "arithmetic", "estimation", "working_backwards"]
MATH_FRAMEWORK_PROMPTS = {
    "algebraic": "You are a math solver. Solve this problem using ALGEBRAIC methods only: set up equations with variables, manipulate them symbolically, and solve for the unknown. Show your algebraic steps.",
    "arithmetic": "You are a math solver. Solve this problem using DIRECT ARITHMETIC only: perform step-by-step numerical calculations without setting up equations. Compute intermediate values explicitly.",
    "estimation": "You are a math solver. Solve this problem using ESTIMATION AND VERIFICATION: first estimate the answer using rounding/approximation, then verify by plugging back in. Show both the estimate and verification.",
    "working_backwards": "You are a math solver. Solve this problem by WORKING BACKWARDS from the answer: start from what you want to find, reverse the operations, and trace back to the given information.",
}

PATTERN_FRAMEWORKS = ["recursive", "formula", "table", "difference"]
PATTERN_FRAMEWORK_PROMPTS = {
    "recursive": "You are a pattern analyst. Find the pattern using a RECURSIVE RULE: express each term as a function of previous terms (e.g., a_n = f(a_{n-1})). State the rule clearly.",
    "formula": "You are a pattern analyst. Find the pattern using a CLOSED-FORM FORMULA: derive an explicit formula a_n = f(n) that gives the nth term directly. Show how you derived it.",
    "table": "You are a pattern analyst. Find the pattern by building a TABLE of values: list the terms, compute differences or ratios between consecutive terms, and identify the pattern from the table.",
    "difference": "You are a pattern analyst. Find the pattern using the METHOD OF DIFFERENCES: compute first differences, second differences, etc., until you find a constant level. Use this to predict the next term.",
}

LOGIC_FRAMEWORKS = ["elimination", "direct_deduction", "case_analysis", "constraint_propagation"]
LOGIC_FRAMEWORK_PROMPTS = {
    "elimination": "You are a logic solver. Solve this using PROCESS OF ELIMINATION: list all possibilities, then systematically eliminate those that violate the given constraints until only one remains.",
    "direct_deduction": "You are a logic solver. Solve this using DIRECT DEDUCTION: start from the given facts and derive new facts step by step using logical inference until you reach the answer.",
    "case_analysis": "You are a logic solver. Solve this using CASE ANALYSIS: break the problem into exhaustive cases, analyze each case separately, and determine which case(s) are consistent with all constraints.",
    "constraint_propagation": "You are a logic solver. Solve this using CONSTRAINT PROPAGATION: list all constraints, then iteratively apply them to narrow down the possibilities for each unknown.",
}

WORD_PROBLEM_FRAMEWORKS = ["diagram", "units_analysis", "simplify_first", "analogous_problem"]
WORD_PROBLEM_FRAMEWORK_PROMPTS = {
    "diagram": "You are a problem solver. Solve this by DRAWING A DIAGRAM (describe it in text): visualize the relationships, label all quantities, and use the diagram to set up the solution.",
    "units_analysis": "You are a problem solver. Solve this using UNITS ANALYSIS: track the units of every quantity, use dimensional analysis to guide which operations to perform, and verify units match.",
    "simplify_first": "You are a problem solver. Solve this by SIMPLIFYING FIRST: identify what information is essential vs. distracting, strip the problem to its core, solve the simplified version, then adjust.",
    "analogous_problem": "You are a problem solver. Solve this by finding an ANALOGOUS SIMPLER PROBLEM: solve a similar but easier version first, identify the method, then apply it to the original.",
}


# ── Task definitions ───────────────────────────────────────────────────

PHASE_A_TASKS: list[Task] = [
    # === Math (algebra/arithmetic) ===
    Task(
        id="math_001", domain="math", difficulty="easy",
        problem="Solve for x: 3x + 7 = 22",
        ground_truth="5",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_002", domain="math", difficulty="easy",
        problem="A store sells notebooks for $4 each and pens for $2 each. Maria buys 3 notebooks and some pens for a total of $20. How many pens did she buy?",
        ground_truth="4",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_003", domain="math", difficulty="easy",
        problem="If 5 workers can paint a house in 8 days, how many days would it take 10 workers to paint the same house?",
        ground_truth="4",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_004", domain="math", difficulty="easy",
        problem="A rectangular garden has a perimeter of 36 meters. If the length is twice the width, what is the area in square meters?",
        ground_truth="72",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_005", domain="math", difficulty="easy",
        problem="A car travels 150 km in 2.5 hours. What is its average speed in km/h?",
        ground_truth="60",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_006", domain="math", difficulty="medium",
        problem="Two trains leave the same station at the same time, traveling in opposite directions. One travels at 80 km/h and the other at 60 km/h. After how many hours will they be 420 km apart?",
        ground_truth="3",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_007", domain="math", difficulty="medium",
        problem="A mixture contains 40% alcohol. How many liters of pure water must be added to 10 liters of this mixture to reduce the alcohol concentration to 25%?",
        ground_truth="6",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_008", domain="math", difficulty="easy",
        problem="What is 15% of 240?",
        ground_truth="36",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_009", domain="math", difficulty="medium",
        problem="The sum of three consecutive even numbers is 78. What is the largest of the three?",
        ground_truth="28",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="math_010", domain="math", difficulty="easy",
        problem="A shirt originally costs $80. It is on sale for 30% off. What is the sale price?",
        ground_truth="56",
        frameworks=MATH_FRAMEWORKS,
        framework_prompts=MATH_FRAMEWORK_PROMPTS,
    ),

    # === Pattern recognition ===
    Task(
        id="pattern_001", domain="pattern", difficulty="easy",
        problem="What is the next number in the sequence: 2, 6, 18, 54, ?",
        ground_truth="162",
        frameworks=PATTERN_FRAMEWORKS,
        framework_prompts=PATTERN_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="pattern_002", domain="pattern", difficulty="easy",
        problem="What is the next number in the sequence: 1, 4, 9, 16, 25, ?",
        ground_truth="36",
        frameworks=PATTERN_FRAMEWORKS,
        framework_prompts=PATTERN_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="pattern_003", domain="pattern", difficulty="medium",
        problem="What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ?",
        ground_truth="21",
        frameworks=PATTERN_FRAMEWORKS,
        framework_prompts=PATTERN_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="pattern_004", domain="pattern", difficulty="easy",
        problem="What is the next number in the sequence: 3, 7, 11, 15, 19, ?",
        ground_truth="23",
        frameworks=PATTERN_FRAMEWORKS,
        framework_prompts=PATTERN_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="pattern_005", domain="pattern", difficulty="medium",
        problem="What is the next number in the sequence: 2, 5, 10, 17, 26, ?",
        ground_truth="37",
        frameworks=PATTERN_FRAMEWORKS,
        framework_prompts=PATTERN_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="pattern_006", domain="pattern", difficulty="easy",
        problem="What is the next number in the sequence: 1, 2, 4, 8, 16, ?",
        ground_truth="32",
        frameworks=PATTERN_FRAMEWORKS,
        framework_prompts=PATTERN_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="pattern_007", domain="pattern", difficulty="medium",
        problem="What is the 10th term of the sequence: 5, 8, 11, 14, 17, ...?",
        ground_truth="32",
        frameworks=PATTERN_FRAMEWORKS,
        framework_prompts=PATTERN_FRAMEWORK_PROMPTS,
    ),

    # === Logic puzzles ===
    Task(
        id="logic_001", domain="logic", difficulty="easy",
        problem="Alice, Bob, and Carol each have a different pet: a cat, a dog, or a fish. Alice doesn't have a cat. Bob doesn't have a dog or a fish. What pet does Carol have?",
        ground_truth="dog",
        frameworks=LOGIC_FRAMEWORKS,
        framework_prompts=LOGIC_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="logic_002", domain="logic", difficulty="easy",
        problem="In a row of 5 houses, the red house is immediately to the left of the blue house. The green house is at one end. The yellow house is next to the green house. The white house is in the middle. What position is the red house in (1-5, left to right)?",
        ground_truth="2",
        frameworks=LOGIC_FRAMEWORKS,
        framework_prompts=LOGIC_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="logic_003", domain="logic", difficulty="medium",
        problem="Three friends - Alex, Blake, and Casey - ordered coffee, tea, and juice (not necessarily in that order). Alex didn't order coffee. The person who ordered tea is sitting between the other two. Blake is sitting on the left end. Casey ordered juice. What did Blake order?",
        ground_truth="tea",
        frameworks=LOGIC_FRAMEWORKS,
        framework_prompts=LOGIC_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="logic_004", domain="logic", difficulty="easy",
        problem="If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Answer yes or no.",
        ground_truth="no",
        frameworks=LOGIC_FRAMEWORKS,
        framework_prompts=LOGIC_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="logic_005", domain="logic", difficulty="medium",
        problem="A farmer has chickens and cows. He counts 30 heads and 80 legs total. How many chickens does he have?",
        ground_truth="20",
        frameworks=LOGIC_FRAMEWORKS,
        framework_prompts=LOGIC_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="logic_006", domain="logic", difficulty="easy",
        problem="Tom is taller than Jerry. Jerry is taller than Spike. Who is the shortest?",
        ground_truth="spike",
        frameworks=LOGIC_FRAMEWORKS,
        framework_prompts=LOGIC_FRAMEWORK_PROMPTS,
    ),

    # === Word problems ===
    Task(
        id="word_001", domain="word_problem", difficulty="easy",
        problem="A tank can be filled by pipe A in 6 hours and by pipe B in 4 hours. If both pipes are opened together, how many hours will it take to fill the tank? Give answer as a decimal.",
        ground_truth="2.4",
        frameworks=WORD_PROBLEM_FRAMEWORKS,
        framework_prompts=WORD_PROBLEM_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="word_002", domain="word_problem", difficulty="medium",
        problem="A boat travels 24 km upstream in 3 hours and returns downstream in 2 hours. What is the speed of the current in km/h?",
        ground_truth="2",
        frameworks=WORD_PROBLEM_FRAMEWORKS,
        framework_prompts=WORD_PROBLEM_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="word_003", domain="word_problem", difficulty="easy",
        problem="If 8 identical machines can produce 480 widgets in 6 hours, how many widgets can 5 machines produce in 4 hours?",
        ground_truth="200",
        frameworks=WORD_PROBLEM_FRAMEWORKS,
        framework_prompts=WORD_PROBLEM_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="word_004", domain="word_problem", difficulty="easy",
        problem="A recipe calls for 2 cups of flour to make 24 cookies. How many cups of flour are needed to make 60 cookies?",
        ground_truth="5",
        frameworks=WORD_PROBLEM_FRAMEWORKS,
        framework_prompts=WORD_PROBLEM_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="word_005", domain="word_problem", difficulty="medium",
        problem="John is twice as old as Mary. In 5 years, John will be 1.5 times as old as Mary. How old is Mary now?",
        ground_truth="10",
        frameworks=WORD_PROBLEM_FRAMEWORKS,
        framework_prompts=WORD_PROBLEM_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="word_006", domain="word_problem", difficulty="medium",
        problem="A store buys items at $40 each and marks them up by 50%. During a sale, it offers a 20% discount. What is the sale price?",
        ground_truth="48",
        frameworks=WORD_PROBLEM_FRAMEWORKS,
        framework_prompts=WORD_PROBLEM_FRAMEWORK_PROMPTS,
    ),
    Task(
        id="word_007", domain="word_problem", difficulty="easy",
        problem="A train 200 meters long passes a pole in 10 seconds. What is the speed of the train in meters per second?",
        ground_truth="20",
        frameworks=WORD_PROBLEM_FRAMEWORKS,
        framework_prompts=WORD_PROBLEM_FRAMEWORK_PROMPTS,
    ),
]
