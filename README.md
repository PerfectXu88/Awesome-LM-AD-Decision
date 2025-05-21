# **Awesome LM for Autonomous Driving Decision-Making**

A comprehensive list of awesome research, resources, and tools for leveraging Large Language Models (LLMs), Vision-Language Models (VLMs), and Vision-Language-Action (VLA) models in autonomous driving decision-making (and motion planning).
Contributions are welcome! 

## Table of Contents

- [Survey papers](#survey-papers)
- [Research Papers](#research-papers) 
    - [Categorization by Model Type](#categorization-by-model-type)
        - [LLM-based Approaches](#llm-based-approaches)
        - [VLM-based Approaches](#vlm-based-approaches)
        - [VLA-based Approaches](#vla-based-approaches)
    - [Categorization by Research Direction](#categorization-by-research-direction) 
        - [End-to-End Driving Models](#end-to-end-driving-models)
        - [Interpretability and Explainable AI (XAI)](#interpretability-and-explainable-ai-xai)
        - [Safety-Critical Decision-Making & Long-Tail Scenarios](#safety-critical-decision-making--long-tail-scenarios) 
        - [Reinforcement Learning for Decision-Making](#reinforcement-learning-for-decision-making)
        - [World Models for Prediction and Planning](#world-models-for-prediction-and-planning)
        - [Hierarchical Planning and Control](#hierarchical-planning-and-control)
    - [Categorization by Application Field in Decision-Making](#categorization-by-application-field-in-decision-making) 
        - [Perception-Informed Decision-Making](#perception-informed-decision-making) 
        - [Behavioral Planning & Prediction](#behavioral-planning--prediction) 
        - [Motion Planning & Trajectory Generation](#motion-planning--trajectory-generation) 
        - [Direct Control Signal Generation](#direct-control-signal-generation)
        - [Human-AI Interaction & Command Understanding](#human-ai-interaction--command-understanding)
    - [Categorization by Technical Route](#categorization-by-technical-route) 
        - [Transformer Architectures & Variants](#transformer-architectures--variants) 
        - [Multimodal Fusion Techniques](#multimodal-fusion-techniques) 
        - [Prompt Engineering](#prompt-engineering-eg-chain-of-thought) 
        - [Knowledge Distillation & Transfer Learning](#knowledge-distillation--transfer-learning) 
- [Datasets and Benchmarks](#datasets-and-benchmarks) 
- [Other Awesome Lists](#other-awesome-lists)



## **Survey papers**

- End-to-end Autonomous Driving: Challenges and Frontiers
- A Superalignment Framework in Autonomous Driving with Large Language Models
- Foundation Models in Robotics: Applications, Challenges, and the Future
- Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis
- A Survey on Vision-Language-Action Models for Embodied AI
- Visual Large Language Models for Generalized and Specialized Applications 
- On the Prospects of Incorporating Large Language Models (LLMs) in Automated Planning and Scheduling
- Foundation Models for Decision Making: Problems, Methods, and Opportunities

## **Research Papers**


### **Categorization by Model Type**

This section further elaborates on papers based on the primary type of foundation model employed for decision-making.


#### **LLM-based Approaches**

LLMs are primarily leveraged for their reasoning, planning, and natural language understanding/generation capabilities to guide autonomous driving decisions. A dominant trend in this area is the hybridization or agentic use of LLMs. Pure LLM-driven control is rare due to challenges in real-time performance, safety assurance, and precise numerical output. Instead, LLMs often function at a strategic or tactical level, acting as a "supervisor," "planner," or "reasoner" that guides more traditional or specialized modules like DRL agents or MPC controllers. This hierarchical system leverages the LLM's strengths in high-level reasoning and contextual understanding while offloading operational, real-time aspects to other components. This approach aims to augment specific parts of the AD stack, particularly those requiring human-like commonsense, reasoning about novel situations, or providing interpretable justifications for actions.



* **TeLL-Drive** : This work proposes a hybrid framework where a "teacher" LLM guides an attention-based "student" Deep Reinforcement Learning (DRL) policy. The LLM utilizes Chain-of-Thought (CoT) reasoning, incorporates risk metrics, and retrieves historical scenarios to produce high-level driving strategies. This approach aims to improve the DRL agent's sample complexity and robustness while ensuring real-time feasibility, a common challenge for LLMs when used in isolation for decision-making.
* **LanguageMPC** : This system employs an LLM as a high-level decision-making component, particularly for complex AD scenarios that demand human commonsense understanding. The LLM processes environmental data through scenario encoding, provides action guidance, and adjusts confidence levels. These high-level decisions are then translated into precise parameters for a low-level Model Predictive Controller (MPC), thereby enhancing interpretability and enabling the system to handle complex maneuvers, including multi-vehicle coordination.
* **Agent-Driver** : This framework positions an LLM as a cognitive agent. The agent has access to a versatile tool library (for perception and prediction tasks), a cognitive memory storing commonsense knowledge and past driving experiences, and a reasoning engine. The reasoning engine is capable of chain-of-thought reasoning, task planning, motion planning, and self-reflection, showcasing a more integrated and sophisticated agentic approach to autonomous driving.
* **LLM for Safety Perspective (Empowering Autonomous Driving with Large Language Models: A Safety Perspective)** : This research explores the integration of LLMs as intelligent decision-makers within the behavioral planning module of AD systems. A key feature is the augmentation of LLMs with a safety verifier shield, which facilitates contextual safety learning. The paper presents studies on an adaptive LLM-conditioned MPC and an LLM-enabled interactive behavior planning scheme using a state machine, demonstrating improved safety metrics.
* **Traffic Regulation Retrieval Agent for Interpretable Autonomous Driving Decision-Making** : This framework introduces an interpretable decision-maker that leverages a Traffic Regulation Retrieval (TRR) Agent, built upon Retrieval-Augmented Generation (RAG). This agent automatically retrieves relevant traffic rules and guidelines from extensive documents. An LLM-powered reasoning module then interprets these rules, differentiates between mandatory regulations and safety guidelines, and assesses actions for legal compliance and safety, enhancing transparency.
* **DiLu (A Knowledge-Driven Approach to Autonomous Driving with Large Language Models)** : DiLu proposes a knowledge-driven framework for autonomous driving that combines Reasoning and Reflection modules within an LLM. This enables the system to make decisions based on common-sense knowledge and to continuously evolve its understanding and strategies through experience.
* **LLM-Planners (General)** : Broader research into LLMs for planning tasks, such as the LLM-Planner for few-shot grounded planning for embodied agents , provides foundational concepts and techniques that can be adapted for AD planning. These works often explore how LLMs can decompose complex tasks and generate sequences of actions.


#### **VLM-based Approaches**

VLMs bring visual understanding to the decision-making process, allowing for richer interpretations of the driving scene and enabling actions based on both visual percepts and linguistic instructions or reasoning. A key challenge for VLMs in AD decision-making is bridging the gap between their often 2D-centric visual-linguistic understanding and the precise, 3D spatio-temporal reasoning essential for safe driving. Many current VLMs are adapted from models pre-trained on large, static 2D image-text datasets , and they can struggle with the dynamic, three-dimensional nature of real-world driving scenarios. This means that while they might excel at describing a scene, their practical performance in actual driving tasks can be concerning. Effective VLMs for AD decision-making will likely need to incorporate stronger 3D visual backbones, improved mechanisms for temporal modeling beyond simple frame concatenation, and potentially integrate more structured environmental representations like Bird's-Eye-View (BEV) maps or scene graphs directly into their reasoning processes.



* **TS-VLM** : This work introduces a lightweight VLM designed for efficient multi-view reasoning in autonomous driving. It features a novel Text-Guided SoftSort Pooling (TGSSP) module that dynamically ranks and fuses visual features from multiple camera views based on the semantics of input queries. This query-aware aggregation aims to improve contextual accuracy and reduce computational overhead, making it more practical for real-time deployment.
* **LightEMMA** : LightEMMA serves as a lightweight, end-to-end multimodal framework and benchmark for evaluating various state-of-the-art commercial and open-source VLMs (such as GPT-4o, Gemini, Claude) in autonomous driving planning tasks. It uses consistent prompting strategies, including Chain-of-Thought (CoT), for structured reasoning and control action prediction on datasets like nuScenes. The findings indicate that despite strong scenario interpretation capabilities, the practical performance of current VLMs in AD tasks remains a concern.
* **DriveGPT4** : This Multimodal Large Language Model (MLLM) is designed for interpretable end-to-end autonomous driving. It processes multi-frame video inputs and textual queries to interpret vehicle actions, provide relevant reasoning, and predict low-level control signals. A bespoke visual instruction tuning dataset aids its capabilities.
* **ADAPT** : While primarily a transformer architecture, ADAPT functions similarly to a VLM by generating natural language narrations and reasoning for driving actions based on video input. It jointly trains the driving captioning task and the vehicular control prediction task, enhancing interpretability in decision-making.
* **LeapAD** : This system uses a VLM for scene understanding, specifically to provide descriptions of critical objects that may influence driving decisions. This visual-linguistic understanding then feeds into a dual-process decision-making module composed of an Analytic LLM and a Heuristic lightweight language model.
* **AlphaDrive** : AlphaDrive is a VLM tailored for high-level planning in autonomous driving. It integrates a Group Relative Policy Optimization (GRPO)-based reinforcement learning strategy with a two-stage reasoning training approach (Supervised Fine-Tuning followed by RL) to boost planning performance and training efficiency.
* **DriveVLM** : This system leverages VLMs for enhanced scene understanding and planning capabilities. It also proposes DriveVLM-Dual, a hybrid system that combines the strengths of DriveVLM with traditional AD pipelines to address VLM limitations in spatial reasoning and computational requirements, particularly for long-tail critical objects.
* **LingoQA** : This project introduces a benchmark and a large dataset (419.9k QA pairs from 28K unique video scenarios) for video question answering specifically in the autonomous driving domain. It focuses on evaluating a VLM's ability to perform reasoning, justify actions, and describe scenes, and proposes the Lingo-Judge metric for evaluation.
* **VLM-E2E** : This framework aims to enhance end-to-end autonomous driving by using VLMs to provide driver attentional cues. It integrates textual representations into Bird's-Eye-View (BEV) features for semantic supervision, enabling the model to learn richer feature representations that capture driver attentional semantics. It also introduces a BEV-Text learnable weighted fusion strategy.
* **VLM-AD** : This method positions VLMs as "teachers" to generate reasoning-based text annotations and structured action labels. These annotations serve as supplementary supervisory signals for training end-to-end AD models, aiming to improve their understanding beyond simple trajectory labels without requiring the VLM at inference time.
* **VL-SAM** : This training-free framework combines a VLM (for generalized object recognition, e.g., recognizing rare objects in AD scenarios) with the Segment-Anything Model (SAM, for generalized object localization). It uses attention maps from the VLM as prompts for SAM to address open-ended object detection and segmentation, which is crucial for robust perception feeding into decision-making systems.


#### **VLA-based Approaches**

VLAs aim to create more generalist agents that can perceive, reason, and act, often in an end-to-end fashion. For autonomous driving, this means models that can take raw sensor data and high-level goals to produce driving actions. While VLAs offer the promise of true end-to-end decision-making by unifying perception, reasoning, and action generation , their application in safety-critical autonomous driving faces a significant hurdle: ensuring the reliability and verifiability of actions generated by these complex, often black-box, generative models. The potential for "hallucinated" or unexpected outputs from generative models is a recurring concern. A major research direction for VLAs in AD will involve developing methods for safety validation, uncertainty quantification, and robust fallback mechanisms. This might include hybrid approaches where VLA outputs are monitored or constrained by traditional safety layers, or novel training paradigms that explicitly optimize for safety and predictability.



* **OpenDriveVLA** : This is an end-to-end VLA model specifically designed for autonomous driving. It generates reliable driving actions conditioned on 3D environmental perception, ego vehicle states, and driver commands. Key methodological contributions include a hierarchical vision-language alignment process to bridge the modality gap between driving visual representations and language embeddings, and an autoregressive agent-env-ego interaction process to ensure spatially and behaviorally informed trajectory planning.
* **NaVILA** : Although developed for legged robot navigation, NaVILA's two-level VLA framework has strong conceptual relevance to autonomous driving. A VLM processes single-view images and outputs mid-level actions in natural language (e.g., "turn right 30 degrees"). These linguistic commands are then executed by a low-level visual locomotion policy trained with RL. This hierarchical approach decouples high-level reasoning from low-level control, allowing the VLA to operate at a lower frequency while the policy handles real-time interactions.
* **3D-VLA** : This work introduces an embodied foundation model that links 3D perception, reasoning, and action through a generative world model. Built on a 3D-LLM, it uses special interaction tokens to engage with the embodied environment. Furthermore, it trains embodied diffusion models, aligned with the LLM, to predict goal images and point clouds, enabling the model to "imagine" future states for planning.
* **QUAR-VLA** : This VLA model is designed for quadruped robots, integrating visual information (first-person camera) and natural language instructions to generate rich control commands, including base velocity, posture, and gait parameters. It leverages a pre-trained VLM fine-tuned on the custom QUARD dataset. Its relevance to AD lies in its methodology for multimodal input processing for generating complex, multi-dimensional actions.
* **CoT-VLA** : This upcoming work (CVPR 2025) aims to incorporate Visual Chain-of-Thought reasoning into VLA models, presumably to enhance the reasoning process that precedes action generation, which could lead to more robust and explainable decisions.
* **General VLA Surveys** : These surveys provide a broad overview of VLA models in robotics and AI, often citing autonomous vehicles as a significant application area. They discuss the evolution from cross-modal learning to generalist agents integrating VLMs, planners, and controllers, highlighting architectural innovations and challenges like real-time control and generalization.

The path to deploying VLAs in real-world AD may involve intermediate steps, such as using them for non-critical decision support, generating synthetic data for training safer traditional models, or operating in "co-pilot" modes. The hierarchical strategy seen in NaVILA, where the VLA provides high-level linguistic commands to a more verifiable low-level controller, could represent a more viable short-term approach for AD, balancing advanced reasoning with dependable execution.

