INITIAL_EXTRACTION_SYS_PROMPT = '''You are an intent extraction assistant. From the [user-assistant conversation history], extract the intents expressed by the user and organize them into a nested JSON structure as follows:
{
  "Type1": {
    "Slot1": "Value (concise and brief)",
    "Slot2": "Value (concise and brief)",
    ...
  },
  "Type2": {
    ...
  }
  ...
}

- "Type1", "Type2": Represent the intent topics.
- "Slot1", "Slot2": Represent the attributes under each intent topic.
- "Value": The value of the attribute, which must be a string. Values must be concise. If too long, shorten or summarize.

Only output JSON that strictly follows this structure. Do NOT add any extra explanations or comments.'''

INITIAL_EXTRACTION_USER_PROMPT = '''user-assistant conversation history:
"""
{chat_history}
"""'''

Utterance_Classification_Sys_Prompt = '''Your task is: Based on [user-assistant conversation history] (with particular focus on the context of the last assistant reply), predict which sentence category the user’s next input might belong to.
## Sentence Category Definitions
- Statement: The user expresses an opinion, feeling, fact, explanation, or feedback, in a calm tone, usually without a question or command nature.
- Question: The user raises a query, seeking information, confirmation, or explanation, usually starting with an interrogative word or with a rising intonation.
- Instruction: The user issues a request, command, or directive for the assistant to perform an action, often in a clear or urgent tone.

## Output Requirements
- For each of Statement, Question, and Instruction sentences, provide reasoning for why the user’s next input might belong to that category.
- The output must strictly follow JSON format.

## Output Format (strict JSON)
[
  {
  "category": "Statement"，
  "reasoning": "<detailed reasoning process>"
  },
  {
  "category": "Question"，
  "reasoning": "<detailed reasoning process>"
  },
  {
  "category": "Instruction"，
  "reasoning": "<detailed reasoning process>"
  }
]'''

Utterance_Classification_User_Prompt = '''[user-assistant conversation history]: {chat_history}'''

Utterance_Classification_GT_Sys_Prompt = '''Your task is: Determine which sentence category the user’s reply to the assistant belongs to.

## Sentence Category Definitions
- Statement: The user expresses an opinion, feeling, fact, explanation, or feedback, in a calm tone, usually without a question or command nature.
- Question: The user raises a query, seeking information, confirmation, or explanation, usually starting with an interrogative word or with a rising intonation.
- Instruction: The user issues a request, command, or directive for the assistant to perform an action, often in a clear or urgent tone.

## Note: Only predict one sentence category.

## Output format (strict JSON):
{
  "reasoning": "<detailed reasoning process>",
  "predicted_category": "<Statement | Question | Instruction>"
}'''

Utterance_Classification_GT_User_Prompt = '''Assistant: {assistant_message}

User: {user_message}'''

Insight_Sys_Prompt ='''You are a "User Input Prediction Expert".  

## Input:
- [user-assistant conversation history]  
- [User intent tree] generated from the conversation  
- The sentence category of the user’s next input (Statement / Question / Instruction)

## Intent Tree Description
- First-level node: Represent the intent topics, such as information inquiry, problem solving, code generation, text editing, translation, creative writing, recommendation, summarization, etc.
- Second-level node: Represent the attributes under each intent topic, such as task description, input content, target language, output format, domain, style requirement, etc.

## Task:
Based on the above input and the assistant’s last reply, infer the user’s possible next input from both the “Mining View” and the “Exploration View”. 
1. Mining View
  - Based on the existing nodes in the intent tree and the assistant’s last reply, infer other potentially interesting or related attributes or changes in attribute values under the current topic that the user may inquire about next.
  - The reasoning can include **two types of continuation** under the same topic:
    - The user may mine a new attribute of the same topic.
    - The user may change an existing attribute’s value while performing a similar operation.
  - During reasoning, explicitly display the intent tree mining path in the format: **Existing Topic Node -> New Attribute Node (or Attribute Value Change)**
2. Exploration View
  - From the overall intent tree and the assistant’s last reply, infer new but closely related topics (not yet mentioned) and their attributes that the user might switch to next.
  - During reasoning, explicitly display the intent tree exploration path in the format: **Existing Topic Node -> New Topic Node-Attribute Node**
  - Note: The exploration path must include a new topic node, not just a new attribute under an existing topic.

## Requirements:
1. Each view must include a **detailed reasoning process**.
2. Each view must output **2 predicted user inputs**, with each prediction corresponding to one intent tree path. 
3. The intent tree path should be displayed **within the reasoning process**, not in the predicted text itself.  
4. The sentence category of all predicted inputs must exactly match the provided user sentence category (Statement / Question / Instruction).

## Output format (strict JSON):
{
  "mining_view": {
    "reasoning": "<detailed reasoning process>",
    "predictions": [
      "<mining prediction 1>",
      "<mining prediction 2>"
    ]
  },
  "explore_view": {
    "reasoning": "<detailed reasoning process>",
    "predictions": [
      "<exploration prediction 1>",
      "<exploration prediction 2>"
    ]
  }
}'''
Insight_User_Prompt = '''user-assistant conversation history:
"""
{chat_history}
"""

user intent tree:   
"""
{intent_tree}
"""

user’s next input sentence category: 
"""
{category}
"""'''

Evaluate_Sys_Prompt = '''You are a "User Intent Similarity Analysis Expert." 
Based on the [user-assistant conversation history], calculate the similarity between the [real next user input] and the [predicted next user input].

# Analysis Objective:
Determine how consistent each predicted input is with the real input in terms of intent and semantic focus.

# Evaluation Dimensions:
1. Keyword Matching: Do both inputs mention the same key entities (e.g., the same product, location, person, event, or concept)?
2. Semantic Intent: Do both inputs aim to achieve the same goal (e.g., requesting information, asking for conditions, seeking help, expressing an opinion, etc.)?

# Similarity Scoring Standard (continuous scale from 0.0 to 1.0, rounded to one decimal place):
- 1.0: Semantics and intent are almost completely identical.
- 0.8–0.9: Highly similar meaning; only minor wording or detail differences.
- 0.4–0.7: Partial overlap in semantics or topic, but intent not fully aligned.
- 0.1–0.3: Only a few shared keywords or topics; overall meaning is different.
- 0.0: Completely unrelated in meaning.

# Only output in the following JSON format:
[
  {"input": "Predicted input", "reason": "Brief explanation (keywords, semantic relation)", "similarity": similarity_score_between_0_and_1},
  {"input": "...", "reason": "...", "similarity": ...},
  ...
]'''

Evaluate_User_Prompt = '''user-assistant conversation history: {context}

real next user input: {label}

predicted next user input: {predict_input}'''


GT_Insight_Path_Sys_Prompt = '''Your task is: Based on [user-assistant conversation history] and the derived [user intent tree], determine whether [the user’s actual next input] belongs to the “Mining View or the “Exploration View,” and provide the corresponding mining or exploration path.

## Intent Tree Description
- First-level node: Represent the main intent **topics**, such as information inquiry, problem solving, code generation, text editing, translation, creative writing, recommendation, summarization, etc.
- Second-level node: Represent the **attributes** under each topic, such as task description, input content, target language, output format, domain, style requirement, etc.

## View Definitions
1. Mining View
  - Based on the existing nodes in the intent tree and the assistant’s last reply, infer other potentially interesting or related attributes or changes in attribute values under the current topic that the user may inquire about next.
2. Exploration View
  - From the overall intent tree and the assistant’s last reply, infer new but closely related topics (not yet mentioned) and their attributes that the user might switch to next.

## Output Requirements
1. Provide a detailed reasoning process explaining why [the user’s actual next input] belongs to either the Mining or Exploration view.
2. Clearly specify the identified view (Mining / Exploration).
3. Based on the identified view, output 1 path that best matches the actual input, along with 1 reasonable but secondary extension paths. Path formats are as follows:
  - Mining Path: Existing Topic Node -> New Attribute Node (or Attribute Value Change).
  - Exploration Path: Existing topic node -> New topic node – Attribute node.
    Requirement: At least one suboptimal extension path in the exploration paths must select a different new topic node to maintain diversity.

## Output Format (Strict JSON):
{
  "insight_reasoning": "<Reasoning process for view judgment>",
  "insight": "Mining or Exploration",
  "path_reasoning": "<Reasoning process for generating paths>",
  "path": [
    {
      "source_node": "<Existing first-level node>",
      "target_node": "<The node that best aligns with the user’s actual next input (meeting view requirements). This node should employ vague or abstract expressions such as “highly rated” or “moderately priced,” avoiding specific numerical values or overly detailed conditions.>"
    },
    {
      "source_node": "<Existing first-level node>",
      "target_node": "<Reasonable extension node (meets view requirement)>"
    }
  ]
}'''
GT_Insight_Path_User_Prompt = '''user-assistant conversation history: {context}

user intent tree: {intent_tree}

the user’s actual next input: {label}'''


Mining_Revise_Sys_Prompt = '''Your initial task was: Based on the [user-assistant conversation history] and the derived [user intent tree], predict the content of the user’s next input.  
At present, you have already obtained the reasoning process.
Your current task is: Following the [intent tree mining path], revise your previous reasoning process. Then, based on the revised reasoning process, the [user-assistant conversation history], and the derived [user intent tree], predict the content of the user’s next input.

## Intent Tree Description
- First-level node: Represent the main intent **topics**, such as information inquiry, problem solving, code generation, text editing, translation, creative writing, recommendation, summarization, etc.
- Second-level node: Represent the **attributes** under each topic, such as task description, input content, target language, output format, domain, style requirement, etc.

## Reasoning Perspective
1. Mining View
  - Based on the existing nodes in the intent tree and the assistant’s last reply, infer other potentially interesting or related attributes or changes in attribute values under the current topic that the user may inquire about next.
  - During reasoning, explicitly display the intent tree mining path in the format: Existing Topic Node -> New Attribute Node (or Attribute Value Change)

## Output Requirements
- Output how you plan to revise the previous reasoning process.
- The revised reasoning path should maintain the same style and logical expression as the original version.
- Predicted content must correspond one-to-one with the [intent tree mining path] in both quantity and content.
- The sentence category of all predicted inputs must exactly match the sentence category (Statement / Question / Instruction) of the user’s next input.

## Output Format (strict JSON):
{
  "thinking": "<How to revise the previous reasoning process>"
  "revised": "<The revised reasoning process>",
  "predictions": [
    "<Prediction 1>",
    "<Prediction 2>"
  ]
}'''

Mining_Revise_User_Prompt = '''user-assistant conversation history: {context}

user intent tree: {intent_tree} 

the sentence category of the user’s next input: {category}

Reasoning process to be revised: {predict_reasoning}

Intent tree mining path: {path}'''

Explore_Revise_Sys_Prompt = '''Your initial task was: Based on the [user-assistant conversation history] and the derived [user intent tree], predict the content of the user’s next input.  
At present, you have already obtained the reasoning process.
Your current task is: Following the [intent tree exploration path], revise your previous reasoning process. Then, based on the revised reasoning process, the [user-assistant conversation history], and the derived [user intent tree], predict the content of the user’s next input.

## Intent Tree Description
- First-level node: Represent the main intent **topics**, such as information inquiry, problem solving, code generation, text editing, translation, creative writing, recommendation, summarization, etc.
- Second-level node: Represent the **attributes** under each topic, such as task description, input content, target language, output format, domain, style requirement, etc.

## Reasoning Perspective
1. Exploration View
  - From the overall intent tree and the assistant’s last reply, infer new but closely related topics (not yet mentioned) and their attributes that the user might switch to next.
  - During reasoning, explicitly display the intent tree exploration path in the format: Existing Topic Node -> New Topic Node-Attribute Node.
  - Note: The exploration path must include a new topic node, not just a new attribute under an existing topic.

## Output Requirements
- Output how you plan to revise the previous reasoning process.  
- The revised reasoning path should maintain the same style and logical expression as the original version.  
- Predicted content must correspond one-to-one with the [intent tree mining path] in both quantity and content. 
- The sentence category of all predicted inputs must exactly match the sentence category (Statement / Question / Instruction) of the user’s next input.

## Output Format (strict JSON):
{
  "thinking": "<How to revise the previous reasoning process>"
  "revised": "<The revised reasoning process>",
  "predictions": [
    "<Prediction 1>",
    "<Prediction 2>"
  ]
}'''

Explore_Revise_User_Prompt = '''user-assistant conversation history: {context}

user intent tree: {intent_tree} 

the sentence category of the user’s next input: {category}

Reasoning process to be revised: {predict_reasoning}

Intent tree exploration path: {path}'''

Incorrect_Path_Sys_Prompt = '''Your task is to generate two incorrect next paths for an intent tree based on the provided [User Intent Tree], [True View], and [True Next Path of the Intent Tree].

## Intent Tree Description
- First-level node: Represent the main intent **topics**, such as information inquiry, problem solving, code generation, text editing, translation, creative writing, recommendation, summarization, etc.
- Second-level node: Represent the **attributes** under each topic, such as task description, input content, target language, output format, domain, style requirement, etc.

## View Definitions
1. Mining View
  - Based on the existing nodes in the intent tree and the assistant’s last reply, infer other potentially interesting or related attributes or changes in attribute values under the current topic that the user may inquire about next.

2. Exploration View
  - From the overall intent tree and the assistant’s last reply, infer new but closely related topics (not yet mentioned) and their attributes that the user might switch to next.

## Output Requirements
You need to generate two incorrect next paths, meaning paths that do not conform to the true next path of the intent tree.
1. If the true view is "Mining View," the two output paths should be incorrect mining paths. The target node format must be: Existing Topic Node -> New Attribute Node (or Attribute Value Change).
2. If the true view is "Exploration View," the two output paths should be incorrect exploration paths, and the two paths must belong to different new topics. The target node format must be: Existing Topic Node -> New Topic Node - Attribute Node.

## Output Format (Strict JSON):
{
  "thinking": "<Your reasoning process for generating the incorrect paths>",
  "path": [
    {
      "source_node": "<Existing first-level node>",
      "target_node": "<Target node (conforming to the view format)>"
    },
    {
      "source_node": "<Existing first-level node>",
      "target_node": "<Target node (conforming to the view format)>"
    }
  ]
}'''


Incorrect_Path_User_Prompt = '''User Intent Tree: {intent_tree}

True View: {gt_insight} view

True Next Path of the Intent Tree: {gt_path}'''

Incorrect_Path_With_Reference_Sys_Prompt = '''Your task is: Based on the [User Intent Tree] and with reference to the [Incorrect User Next Input], generate two incorrect next-step paths for the intent tree.

## Intent Tree Description
- First-level node: Represent the main intent **topics**, such as information inquiry, problem solving, code generation, text editing, translation, creative writing, recommendation, summarization, etc.
- Second-level node: Represent the **attributes** under each topic, such as task description, input content, target language, output format, domain, style requirement, etc.

## View Definitions
1. Mining View
  - Based on the existing nodes in the intent tree and the assistant’s last reply, infer other potentially interesting or related attributes or changes in attribute values under the current topic that the user may inquire about next.

2. Exploration View
  - From the overall intent tree and the assistant’s last reply, infer new but closely related topics (not yet mentioned) and their attributes that the user might switch to next.

## Definition of Incorrect Paths
- The incorrect paths should be semantically reasonable, but not consistent with the ground-truth next-step path.
- The deviation can occur at either the topic level, the attribute level, or both.

## Output Requirements
1. Give priority to the [Incorrect User Next Input] as a reference to generate the incorrect next-step paths of the intent tree.
2. Output the paths following the **ground-truth view format**:
  - For **Mining View**: `<existing topic node> -> <new attribute node>`
  - For **Exploration View**: `<existing topic node> -> <new topic node>-<attribute node>`
3. Think carefully before producing the output.

## Output Format (Strict JSON):
{
  "thinking": "<reasoning process for generating incorrect paths>",
  "path": [
    {
      "source_node": "<Existing first-level node>",
      "target_node": "<Target node (following the ground-truth view format)>"
    },
    {
      "source_node": "<Existing first-level node>",
      "target_node": "<Target node (following the ground-truth view format)>"
    }
  ]
}'''

Incorrect_Path_With_Reference_User_Prompt = '''User Intent Tree: {intent_tree}

Ground-Truth view: {gt_insight} view

Ground-Truth Next-Step Path in Intent Tree: {gt_path}

Incorrect User Next Input: {error_user_input}'''

INFER_PROMPT = '''Your task is to infer and analyze based on the [user-assistant conversation history] and predict the user’s most likely next input.

# Step 1: Build the User Intent Tree
Extract the user’s intents from the conversation and organize them as a tree. Use the following output format:
{{
  "Type1": {{
    "Slot1": "Value (concise and brief)",
    "Slot2": "Value (concise and brief)",
    ...
  }},
  "Type2": {{
    ...
  }}
  ...
}}

- "Type1", "Type2": Represent the intent topics.
- "Slot1", "Slot2": Represent the attributes under each intent topic.
- "Value": The value of the attribute, which must be a string. Values must be concise. If too long, shorten or summarize.

# Step 2: Sentence Category Inference for the User’s Next Input
- Based on the conversation history (especially the assistant’s most recent reply), predict the sentence category of the user’s next input—choose only one of the three: Statement, Question, or Instruction.
   - Statement: The user expresses an opinion, feeling, fact, explanation, or feedback, in a calm tone, usually without a question or command nature.
   - Question: The user raises a query, seeking information, confirmation, or explanation, usually starting with an interrogative word or with a rising intonation.
   - Instruction: The user issues a request, command, or directive for the assistant to perform an action, often in a clear or urgent tone.
- First output a detailed reasoning process, then output the predicted sentence category.
- The sentence category of all predicted inputs must exactly match the predicted sentence category (Statement / Question / Instruction).

# Step 3: Predict the User’s Most Likely Next Input
Based on [user-assistant conversation history](especially the assistant’s last reply), the [User Intent Tree], and the [predicted sentence category], infer the user's potential next-turn input from both “Mining” and “Exploration” Views. Each view predicts two possible inputs (language consistent with historical conversations).
1. Mining View
  - Based on the existing nodes in the intent tree and the assistant’s last reply, infer other potentially interesting or related attributes or changes in attribute values under the current topic that the user may inquire about next.
  - The reasoning can include **two types of continuation** under the same topic:
    - The user may mine a new attribute of the same topic.
    - The user may change an existing attribute’s value while performing a similar operation.
  - During reasoning, explicitly display the intent tree mining path in the format: Existing Topic Node -> New Attribute Node (or Attribute Value Change)
2. Exploration View
  - From the overall intent tree and the assistant’s last reply, infer new but closely related topics (not yet mentioned) and their attributes that the user might switch to next.
  - During reasoning, explicitly display the intent tree exploration path in the format: Existing Topic Node -> New Topic Node-Attribute Node.
  - Note: The exploration path must include a new topic node, not just a new attribute under an existing topic.

# Output Format:
<think>Detailed reasoning for each step</think>
<predict>Prediction #1</predict>
<predict>Prediction #2</predict>
<predict>Prediction #3</predict>
<predict>Prediction #4</predict>

--Input Information--
[user-assistant conversation history]: 
```
{chat_history}
```'''

OUTPUT_PROMPT = '''<think>First, I need to extract user intent from the historical conversation between the user and the assistant and represent it using a tree structure.
[User Intent Tree]: ```
{intent_tree}
```

Then, I need to analyze the sentence type of the user's next input.
[Analysis of the sentence category for the user's next input]: {utterance_category_reasoning}
Therefore, the sentence category for the user's next input is most likely {utterance_category}.

Finally, I need to infer potential inputs that may interest the user based on the intent tree, considering both mining and exploration views.
[Mining View Analysis]: {mining_reasoning}
[Exploration View Analysis]: {exploration_reasoning}
</think>
The user’s next input is most likely: 
{predictions}'''