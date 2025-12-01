## 💡 Thinking Cobra: Extending Cobra VLM with reasoning

**Authors' Contact Information:** András Ferenczy, agf64, Hungary, Isaac Steinberg, ies22, USA, Matteo Perona, mp2334, USA.
Manuscript submitted to ACM.

### Abstract

[cite_start]We investigate whether integrating explicit reasoning capabilities can enhance the multimodal performance of small-scale vision-language models[cite: 26]. [cite_start]Building on Cobra, a Mamba-based VLM designed for efficient inference, we introduce a two-pass scratchpad reasoning mode that generates intermediate analytical traces before producing final captions, without modifying the underlying model architecture[cite: 27].

[cite_start]We further explore supervised enhancement of reasoning by fine-tuning Cobra on LLAVA-CoT-100k, a dataset containing 100,000 visual question-answering samples with structured reasoning annotations spanning visual interpretation, logical reasoning, and conclusion generation[cite: 28].

[cite_start]Our evaluation protocol employs the COCO Caption dataset with BLEU metrics as the primary benchmark, supplemented by embedding-based semantic similarity to capture meaning beyond n-gram overlap[cite: 29]. [cite_start]Preliminary baseline results yield BLEU scores of 0.47, 0.31, 0.20, and 0.13 for n-gram orders 1 through 4[cite: 30].

[cite_start]By systematically comparing baseline Cobra, scratchpad-prompted variants, and models fine-tuned on reasoning data, we aim to isolate the contributions of inference-time prompting versus learned reasoning patterns[cite: 30]. [cite_start]As a stretch goal, we propose applying Group Relative Policy Optimization (GRPO) to further enhance reasoning through reinforcement learning, following recent advances in reasoning-focused language models[cite: 31]. [cite_start]This work provides insights into how far compact, efficient VLMs can be improved through reasoning mechanisms while preserving their computational advantages[cite: 32].

---

[cite_start]**CCS Concepts:** Computer systems organization Neural networks: General and reference Experimentation[cite: 35, 36].
[cite_start]**Additional Key Words and Phrases:** vision-language models, multimodal reasoning, scratchpad, Mamba architecture, image captioning, COCO dataset, BLEU, LLAVA-COT[cite: 40].

---

### 1 What is the problem that you are investigating? What paper/papers are you basing your investigation on?

[cite_start]We propose to investigate whether integrating reasoning capabilities into small-scale vision-language models can significantly improve their multimodal performance[cite: 55].

* [cite_start]Building on *Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference* [cite: 56][cite_start], a Mamba-based efficient VLM, we will extend it with explicit chain-of-thought prompting or a latent reasoning module inspired by recent reasoning approaches in LLMs, adapting these techniques to Mamba's architecture[cite: 56].
* [cite_start]We want to explore whether reasoning mechanisms can enhance small, efficient VLMs without sacrificing their computational advantages[cite: 57].
* [cite_start]By comparing baseline Cobra against chain-of-thought or latent reasoning variants, we aim to examine performance gains and evaluate the transferability of reasoning techniques across model architectures[cite: 58, 60].

### 2 A formal description of your problem, including the data and evaluation metrics you will be using.

[cite_start]Our core research question is: does adding reasoning improve the performance of small VLMs, and if yes, by how much? [cite: 67] [cite_start]Our goal is to extend the Cobra VLM implementation with a "reasoning-style" component and benchmark it with the baseline implementation to quantify its effect[cite: 68, 70].

#### 2.1 Dataset

[cite_start]For training and evaluation, we will use the **COCO caption**, a multimodal (text and image) dataset[cite: 79].

* [cite_start]The dataset contains almost **60,000 images**, each with the prompt question and label captions[cite: 80].
* [cite_start]The dataset is split into a **training (5,000)** and **evaluation (40,700)** set[cite: 81].

**Fig. 1. Example from the dataset**
[cite_start] [cite: 99]

#### 2.2 Evaluation

[cite_start]After training the model with and without an added reasoning part, we will use the **BiLingual Evaluation Understudy (BLEU)**, a standard multimodal benchmark developed by IBM in 2002[cite: 100].

* [cite_start]BLEU measures how closely machine-generated captions overlap with reference human-written captions by computing n-gram precision at multiple levels[cite: 138, 139].
* [cite_start]For each n-gram size ($n=1, 2, 3, 4$), BLEU calculates the proportion of n-grams in the generated caption that appear in the reference captions[cite: 139].
* The final BLEU score is computed as:
$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} \frac{1}{N} \log p_{n}\right)
$$
* [cite_start]**BP (brevity penalty)** acts as a corrective factor in scoring translations that are shorter than the reference text[cite: 143]. [cite_start]It is given by $\min(1, \text{reference\_length}/\text{translated\_length})$, where $\text{reference\_length}$ is the total word count in the reference text and $\text{translated\_length}$ is the total word count of the generated output[cite: 144].
* [cite_start]$p_{n}$, the modified n-gram precision, quantifies the ratio of n-grams in the generated output that also appear in the reference text[cite: 145]. [cite_start]It is calculated by dividing the number of shared n-grams between the candidate and reference by the total number of n-grams in the candidate[cite: 146].
* [cite_start]We chose BLEU over ROUGE or CIDEr because we concluded it was the most comprehensive and easy-to-use benchmark for this case [cite: 147][cite_start], and we agree with its creators that it is the most human-like evaluation metric for our study[cite: 148, 149].
* [cite_start]However, a disadvantage of BLEU is that since it compares n-grams, it does not capture the meaning itself[cite: 154].

### 3 What method are you using? You should provide an explicit formal description of the method.

[cite_start]We implement a **two-pass scratchpad reasoning mode** as an inference-time extension of Cobra, based on the concept of scratchpads for intermediate computation in language models[cite: 158, 376]. [cite_start]This approach introduces intermediate reasoning traces without modifying any component of Cobra's underlying architecture[cite: 159].

* [cite_start]**Baseline Cobra:** Given an image and a captioning instruction (e.g., "Describe the image"), the baseline model generates a caption directly in a single inference pass[cite: 160].
* [cite_start]**Scratchpad Mode (Two Sequential Stages):** Caption generation is decomposed into two sequential stages[cite: 179].
    1.  [cite_start]**First Stage:** The model is prompted with a reasoning instruction to analyze the image step-by-step, while explicitly forbidding it from producing the final caption[cite: 180]. [cite_start]This produces an intermediate reasoning trace enumerating salient objects, attributes, and spatial relationships[cite: 181].
    2.  [cite_start]**Second Stage:** The model is prompted again with both the original input **and** its own reasoning trace and is instructed to produce only the final caption[cite: 182].
* [cite_start]The difference between modes arises solely from the inference protocol, as both stages use the same model parameters and architecture[cite: 183]. [cite_start]This enables a clean comparison between baseline captioning and scratchpad-enhanced captioning using identical model weights, COCO images, and BLEU-based evaluation metrics[cite: 184, 186].

#### 3.1 Fine-tuning with Structured Reasoning Data

[cite_start]To enhance Cobra's reasoning capabilities beyond inference-time prompting, we fine-tune the model on the **LLaVA-CoT-100k dataset**[cite: 222, 374].

* [cite_start]This dataset provides structured reasoning annotations across diverse visual question-answering samples, which is well-suited for teaching models to perform autonomous multistage reasoning[cite: 223, 224].
* [cite_start]Unlike the COCO caption dataset used for initial benchmarking, LLaVA-CoT-100k explicitly annotates intermediate reasoning steps, including visual interpretation, logical reasoning, and conclusion generation[cite: 225].
* [cite_start]The dataset contains **100,000 samples** from various visual question-answering sources, each annotated with a structured reasoning chain that systematically analyzes visual inputs and arrives at conclusions[cite: 226].

[cite_start]We incorporate this dataset into our training pipeline in two ways[cite: 228]:

1.  [cite_start]**Reasoning-Enhanced Variant:** We continue training the baseline Cobra model on LLaVA-CoT-100k, optimizing it to predict both the intermediate reasoning steps and final answers[cite: 229]. [cite_start]This creates a reasoning-enhanced variant of Cobra that has learned to engage in structured analysis through supervised learning[cite: 230].
2.  [cite_start]**Synergistic Effect:** We evaluate whether this fine-tuned model produces better reasoning traces when used with our two-pass scratchpad approach, potentially creating a synergistic effect where learned reasoning patterns reinforce prompted reasoning behavior[cite: 231].

[cite_start]By comparing three variants (baseline Cobra, Cobra with scratchpad prompting only, and Cobra fine-tuned on LLaVA-CoT-100k with scratchpad prompting), we can isolate the contributions of inference-time reasoning versus learned reasoning capabilities[cite: 232]. [cite_start]We expect the fine-tuned model to generate more coherent and informative reasoning traces in the first pass, which should translate to improved caption quality (measured by BLEU scores and semantic similarity metrics on the COCO evaluation set) in the second pass[cite: 233].

#### 3.2 Stretch Goal: Reinforcement Learning with GRPO

[cite_start]As a stretch goal, we propose applying **Group Relative Policy Optimization (GRPO)** to further enhance Cobra's reasoning capabilities[cite: 248].

* [cite_start]GRPO, introduced by [cite: 378] [cite_start]and employed in the development of DeepSeek-R1 [cite: 372][cite_start], offers a computationally efficient alternative to traditional actor-critic reinforcement learning methods by eliminating the need for a separate critic model[cite: 249].
* [cite_start]The GRPO algorithm optimizes the policy model by sampling a group of outputs for each question and using group-based advantage estimation[cite: 250].
* For a given image captioning query $q$, GRPO samples $G$ outputs $\{o_{1},o_{2},...,o_{G}\}$ from the current policy $\pi_{\theta}$ and optimizes the following objective:
$$
J_{\text{GRPO}} (\theta) = \mathop{\mathbb{E}}_{q \sim \mathcal{D}} \left[ \sum_{i=1}^{G} \min \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \right) A_i \right) - \beta \text{DKL}(\pi_{\theta} || \pi_{\text{ref}}) \right]
$$
* [cite_start]The advantage $A_{i}$ is computed using group normalization[cite: 267]:
$$
A_{i}=\frac{r_{i}-mean(\{r_{1},r_{2},...,r_{G}\})}{std(\{r_{1},r_{2},...,r_{G}\})}
$$
* [cite_start]We would design a reward function for our caption generation task comprising both **accuracy** and **format** components[cite: 296]. [cite_start]The accuracy reward would evaluate semantic alignment using embedding-based similarity metrics or neural reward models[cite: 297]. [cite_start]The format reward would encourage the model to produce well-structured outputs that include reasoning traces when appropriate[cite: 298].
* [cite_start]The GRPO-trained Cobra variant would represent our most sophisticated model, combining learned reasoning patterns from supervised fine-tuning with policy optimization driven by caption quality rewards[cite: 301].
* [cite_start]However, GRPO is a stretch goal due to substantial computational challenges, including increased inference costs from sampling multiple outputs, and the difficulty of designing robust reward models for caption quality[cite: 303, 304, 305].

### 4 Preliminary Results

[cite_start]As of the milestone submission date, we benchmarked Cobra with BLEU on the COCO dataset, obtaining the following results[cite: 312]:

* [cite_start]**BLEU-1:** 0.4702 [cite: 314]
* [cite_start]**BLEU-2:** 0.3110 [cite: 316]
* [cite_start]**BLEU-3:** 0.1992 [cite: 319]
* [cite_start]**BLEU-4:** 0.1266 [cite: 345]

[cite_start]We anticipate improvements from adding a reasoning component to the VLM[cite: 347].

* [cite_start]Because BLEU has known limitations, we will also consider evaluating caption quality using a simple **embedding-based similarity metric**[cite: 348].
* [cite_start]This involves embedding all human COCO captions using a fixed text encoder (e.g., Sentence-BERT or the CLIP text model) and averaging the embeddings for each image[cite: 349].
* [cite_start]For each caption generated by Cobra, we will compute its embedding with the same encoder and measure the **cosine similarity** to the reference embedding[cite: 350].
* [cite_start]Averaging these scores across the dataset provides a semantic alignment metric that captures meaning rather than exact word overlap, offering a useful complement to BLEU[cite: 351].

[cite_start]**Example showing the generated caption with reference captions for the same image as on Fig. 1**[cite: 352]:

* [cite_start]**Generated caption:** A black and silver motorcycle parked on a grassy area with a white garage door and green trees in the background[cite: 353].
* **Reference captions:**
    * [cite_start]A black Honda motorcycle parked in front of a garage[cite: 355].
    * [cite_start]A Honda motorcycle parked in a grass driveway[cite: 355].
    * [cite_start]A black Honda motorcycle with a dark burgundy seat[cite: 356].
    * [cite_start]A motorcycle parked on the gravel in front of a garage[cite: 357].
    * [cite_start]A motorcycle with its brake extended standing outside[cite: 357].

### References

* [1] Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Li, J., Cai, J., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R., Jin, R., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948 [cs.LG]*.
* [2] Wu, Z., Li, H., Song, Y., Sun, L., Yuan, L., Xu, G., & Jin, P. (2024). LLaVA-COT: Let Vision Language Models Reason Step-by-Step. *arXiv:2411.10440*.
* [3] Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H., Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., Sutton, C., & Odena, A. (2021). Show Your Work: Scratchpads for Intermediate Computation with Language Models. *arXiv:2112.00114 [cs.LG]*.
* [4] Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., Guo, D., Shao, Z., & Wang, P. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300 [cs.LG]*.