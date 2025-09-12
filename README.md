<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---
Join us at the [PyTorch Conference, October 22-23](https://events.linuxfoundation.org/pytorch-conference/) and [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco for our latest updates on vLLM and to meet the vLLM team! Register now for the largest vLLM community events of the year!

---

*Latest News* 🔥

- [2025/08] We hosted [vLLM Shenzhen Meetup](https://mp.weixin.qq.com/s/k8ZBO1u2_2odgiKWH_GVTQ) focusing on the ecosystem around vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Ua2SVKVSu-wp5vou_6ElraDt2bnKhiEA).
- [2025/08] We hosted [vLLM Singapore Meetup](https://www.sginnovate.com/event/vllm-sg-meet). We shared V1 updates, disaggregated serving and MLLM speedups with speakers from Embedded LLM, AMD, WekaIO, and A*STAR. Please find the meetup slides [here](https://drive.google.com/drive/folders/1ncf3GyqLdqFaB6IeB834E5TZJPLAOiXZ?usp=sharing).
- [2025/08] We hosted [vLLM Shanghai Meetup](https://mp.weixin.qq.com/s/pDmAXHcN7Iqc8sUKgJgGtg) focusing on building, developing, and integrating with vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1OvLx39wnCGy_WKq8SiVKf7YcxxYI3WCH).
- [2025/05] vLLM is now a hosted project under PyTorch Foundation! Please find the announcement [here](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/).
- [2025/01] We are excited to announce the alpha release of vLLM V1: A major architectural upgrade with 1.7x speedup! Clean code, optimized execution loop, zero-overhead prefix caching, enhanced multimodal support, and more. Please check out our blog post [here](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).

<details>
<summary>Previous News</summary>

- [2025/08] We hosted [vLLM Korea Meetup](https://luma.com/cgcgprmh) with Red Hat and Rebellions! We shared the latest advancements in vLLM along with project spotlights from the vLLM Korea community. Please find the meetup slides [here](https://drive.google.com/file/d/1bcrrAE1rxUgx0mjIeOWT6hNe2RefC5Hm/view).
- [2025/08] We hosted [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/dgkWg1WFpWGO2jCdTqQHxA) focusing on large-scale LLM deployment! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF) and the recording [here](https://www.chaspark.com/#/live/1166916873711665152).
- [2025/05] We hosted [NYC vLLM Meetup](https://lu.ma/c1rqyf1f)! Please find the meetup slides [here](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?usp=sharing).
- [2025/04] We hosted [Asia Developer Day](https://www.sginnovate.com/event/limited-availability-morning-evening-slots-remaining-inaugural-vllm-asia-developer-day)! Please find the meetup slides from the vLLM team [here](https://docs.google.com/presentation/d/19cp6Qu8u48ihB91A064XfaXruNYiBOUKrBxAmDOllOo/edit?usp=sharing).
- [2025/03] We hosted [vLLM x Ollama Inference Night](https://lu.ma/vllm-ollama)! Please find the meetup slides from the vLLM team [here](https://docs.google.com/presentation/d/16T2PDD1YwRnZ4Tu8Q5r6n53c5Lr5c73UV9Vd2_eBo4U/edit?usp=sharing).
- [2025/03] We hosted [the first vLLM China Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg)! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1REHvfQMKGnvz6p3Fd23HhSO4c8j5WPGZV0bKYLwnHyQ/edit?usp=sharing).
- [2025/03] We hosted [the East Coast vLLM Meetup](https://lu.ma/7mu4k4xx)! Please find the meetup slides [here](https://docs.google.com/presentation/d/1NHiv8EUFF1NLd3fEYODm56nDmL26lEeXCaDgyDlTsRs/edit#slide=id.g31441846c39_0_0).
- [2025/02] We hosted [the ninth vLLM meetup](https://lu.ma/h7g3kuj9) with Meta! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1jzC_PZVXrVNSFVCW-V4cFXb6pn7zZ2CyP_Flwo05aqg/edit?usp=sharing) and AMD [here](https://drive.google.com/file/d/1Zk5qEJIkTmlQ2eQcXQZlljAx3m9s7nwn/view?usp=sharing). The slides from Meta will not be posted.
- [2025/01] We hosted [the eighth vLLM meetup](https://lu.ma/zep56hui) with Google Cloud! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1epVkt4Zu8Jz_S5OhEHPc798emsYh2BwYfRuDDVEF7u4/edit?usp=sharing), and Google Cloud team [here](https://drive.google.com/file/d/1h24pHewANyRL11xy5dXUbvRC9F9Kkjix/view?usp=sharing).
- [2024/12] vLLM joins [pytorch ecosystem](https://pytorch.org/blog/vllm-joins-pytorch)! Easy, Fast, and Cheap LLM Serving for Everyone!
- [2024/11] We hosted [the seventh vLLM meetup](https://lu.ma/h0qvrajz) with Snowflake! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit?usp=sharing), and Snowflake team [here](https://docs.google.com/presentation/d/1qF3RkDAbOULwz9WK5TOltt2fE9t6uIc_hVNLFAaQX6A/edit?usp=sharing).
- [2024/10] We have just created a developer slack ([slack.vllm.ai](https://slack.vllm.ai)) focusing on coordinating contributions and discussing features. Please feel free to join us there!
- [2024/10] Ray Summit 2024 held a special track for vLLM! Please find the opening talk slides from the vLLM team [here](https://docs.google.com/presentation/d/1B_KQxpHBTRa_mDF-tR6i8rWdOU5QoTZNcEg2MKZxEHM/edit?usp=sharing). Learn more from the [talks](https://www.youtube.com/playlist?list=PLzTswPQNepXl6AQwifuwUImLPFRVpksjR) from other vLLM contributors and users!
- [2024/09] We hosted [the sixth vLLM meetup](https://lu.ma/87q3nvnh) with NVIDIA! Please find the meetup slides [here](https://docs.google.com/presentation/d/1wrLGwytQfaOTd5wCGSPNhoaW3nq0E-9wqyP7ny93xRs/edit?usp=sharing).
- [2024/07] We hosted [the fifth vLLM meetup](https://lu.ma/lp0gyjqr) with AWS! Please find the meetup slides [here](https://docs.google.com/presentation/d/1RgUD8aCfcHocghoP3zmXzck9vX3RCI9yfUAB2Bbcl4Y/edit?usp=sharing).
- [2024/07] In partnership with Meta, vLLM officially supports Llama 3.1 with FP8 quantization and pipeline parallelism! Please check out our blog post [here](https://blog.vllm.ai/2024/07/23/llama31.html).
- [2024/06] We hosted [the fourth vLLM meetup](https://lu.ma/agivllm) with Cloudflare and BentoML! Please find the meetup slides [here](https://docs.google.com/presentation/d/1iJ8o7V2bQEi0BFEljLTwc5G1S10_Rhv3beed5oB0NJ4/edit?usp=sharing).
- [2024/04] We hosted [the third vLLM meetup](https://robloxandvllmmeetup2024.splashthat.com/) with Roblox! Please find the meetup slides [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] We hosted [the second vLLM meetup](https://lu.ma/ygxbpzhl) with IBM! Please find the meetup slides [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2023/10] We hosted [the first vLLM meetup](https://lu.ma/first-vllm-meetup) with a16z! Please find the meetup slides [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/08] We would like to express our sincere gratitude to [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) for providing a generous grant to support the open-source development and research of vLLM.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

</details>

---

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  151.32
Total input tokens:                      100608
Total generated tokens:                  127697
Request throughput (req/s):              6.61
Output token throughput (tok/s):         843.87
Total Token throughput (tok/s):          1508.72
---------------Time to First Token----------------
Mean TTFT (ms):                          66172.22
Median TTFT (ms):                        56837.92
P99 TTFT (ms):                           144176.81
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          238.06
Median TPOT (ms):                        265.18
P99 TPOT (ms):                           369.99
---------------Inter-token Latency----------------
Mean ITL (ms):                           240.68
Median ITL (ms):                         241.09
P99 ITL (ms):                            2159.38
==================================================

```

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  152.42
Total input tokens:                      100608
Total generated tokens:                  127700
Request throughput (req/s):              6.56
Output token throughput (tok/s):         837.80
Total Token throughput (tok/s):          1497.85
---------------Time to First Token----------------
Mean TTFT (ms):                          66791.51
Median TTFT (ms):                        55423.42
P99 TTFT (ms):                           145447.24
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          241.15
Median TPOT (ms):                        259.23
P99 TPOT (ms):                           371.23
---------------Inter-token Latency----------------
Mean ITL (ms):                           243.91
Median ITL (ms):                         260.70
P99 ITL (ms):                            2068.21
==================================================
```

```
ChatCompletion(id='chatcmpl-4268cf6911374e239475864d1b0c27a8', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='<think>Got it, let\'s analyze the problem. The question is about a c
ube losing a vertex (corner) and what the resulting shape is, then identifying which numbered figure (1-4) matches. \n\nFirst, a cube has 8 vertices. When a corner (vertex) is removed, we\'re essentially cutting off a corner, which
is a tetrahedral cut. The resulting shape should be a polyhedron. Let\'s think about the faces: originally, a cube\'s vertex is where three faces meet. Removing a vertex would create a new triangular face where the corner was, and t
he original adjacent faces would be modified. \n\nNow, let\'s look at the figures. Figure 3 and 4 have larger triangular faces, while 1 and 2 have smaller triangular faces. When you cut off a vertex of a cube, the new face should be
 a triangle, and the original cube\'s faces near the cut would have a new edge (a triangle\'s edge). Wait, the shape after removing a vertex is a polyhedron with how many faces? Let\'s count: a cube has 6 faces. When you remove a ve
rtex, you replace one vertex with a triangular face, so the number of faces becomes 6 - 1 + 1 (the new triangular face) = 6? Wait, no. Wait, when you cut a corner (a vertex), you\'re creating a new face. Each vertex removal adds a f
ace. So original cube has 6 faces, 8 vertices. Each vertex removal adds 1 face, so 6 + 1 = 7 faces? Wait, maybe better to visualize. The shape after cutting off a corner of a cube is a polyhedron called a "truncated cube" part, but
maybe more specifically, the shape has a triangular face where the corner was, and the other faces are quadrilaterals or triangles? Wait, let\'s check the figures. Figure 3 and 4 have a large triangular face, while 1 and 2 have smal
ler triangular faces. The key is that when you remove a vertex, the resulting shape has a triangular face (the new face) and the original cube\'s adjacent faces are each modified by a triangle. Wait, let\'s think about the number of
 edges and faces. Alternatively, the shape after removing a vertex from a cube is a polyhedron with 7 faces? Wait, no. Wait, the correct shape when you cut off a corner of a cube is a polyhedron with 5 faces? Wait, no. Wait, let\'s
look at the figures. Figure 3 and 4 have a large triangular face, which would be the new face created by cutting the corner. The other faces are quadrilaterals or triangles? Wait, the original cube has 6 faces. When you cut a corner
, you\'re removing a vertex, which is connected to 3 edges. Cutting the corner would create a triangular face (3 edges) and modify the 3 adjacent faces (each of those faces would have a new edge, a triangle\'s edge). So each of the
3 adjacent faces becomes a pentagon? Wait, no. Wait, if you have a cube, and you cut a corner, the intersection is a triangle, so the new face is a triangle, and the original three faces (which met at the corner) each have a triangu
lar cut, so each of those three faces becomes a quadrilateral? Wait, no. Let\'s think of a cube with a corner (vertex) removed. The resulting shape is a polyhedron with 5 faces? Wait, no. Wait, let\'s count the faces in figure 3: it
 has a large triangular face and three quadrilateral faces? Wait, figure 3 shows a shape with a large triangular face and the other faces—wait, the image shows figure 3 as having a large gray triangular face, and the other faces are
 white. Let\'s check the structure. When you remove a vertex from a cube, the resulting shape has a triangular face (the one where the vertex was) and the other faces are quadrilaterals. Wait, maybe the correct figure is 3 or 4. Wai
t, let\'s think about the orientation. The key is that the shape after removing a vertex has a triangular face, and the number of faces. Alternatively, the shape is a tetrahedron-like addition but on a cube. Wait, the correct answer
 is likely figure 3 or 4. Wait, let\'s recall that when you cut off a corner of a cube, the resulting polyhedron has a triangular face, and the original cube\'s faces near the cut are each reduced by a triangle, making them quadrila
terals. So the shape would have 5 faces: the new triangular face and 4 quadrilaterals? No, wait, a cube has 6 faces. When you remove a vertex, you remove one vertex, and the new face is added, so 6 - 1 + 1 = 6 faces? Hmm, maybe I\'m
 overcomplicating. Let\'s look at the figures. Figure 1 and 2 have small triangular faces (the gray area is a small triangle), while 3 and 4 have large triangular faces. When you remove a corner (vertex) of a cube, the cut creates a
 larger triangular face, so the gray area should be a larger triangle, which would be figure 3 or 4. Now, which one is it? Let\'s think about the perspective. The shape after cutting a corner should have a triangular face that\'s a
new face, and the other faces are the original faces minus the corner. So the gray area in figure 3 is a large triangle, which would be the new face, and the other faces are the remaining parts. So the correct number is 3 or 4? Wait
, maybe the answer is 3 or 4, but let\'s check again. Wait, the problem says "a cube loses one vertex after a \'corner\' is removed. This geometric shape is ___". The shape after removing a corner (vertex) of a cube is a polyhedron
called a "truncated cube" part, but more precisely, it\'s a shape with a triangular face. Among the options, figure 3 and 4 have large triangular faces. Let\'s check the number of edges. The original cube has 12 edges. After removin
g a vertex, how many edges does the new shape have? Each vertex removal adds 3 edges (the edges of the new triangular face), but maybe that\'s not helpful. Alternatively, the correct figure is 3. Wait, no, let\'s think of the gray a
rea. In figure 3, the gray area is a large triangle covering a significant portion, while in figure 4, it\'s also a large triangle. Wait, maybe the key is that when you remove a corner, the resulting shape has a triangular face, and
 the figure with the largest triangular face is the one. But I need to pick between 3 and 4. Wait, maybe the answer is 3. Wait, no, let\'s do a step-by-step. Step 1: A cube has 8 vertices. Removing one vertex means the shape now has
 7 vertices. Step 2: When a vertex is removed (cut off), the shape gains a new face (a triangle, since the intersection of the cut is a triangle). Step 3: The original cube had 6 faces; after removing a vertex, one face is modified,
 and a new face is added, so total faces are 6 - 1 + 1 = 6? Wait, no, each vertex removal adds a face. So 6 + 1 = 7 faces? Hmm. Step 4: Looking at the figures, figure 3 has a large triangular face (new face) and the other faces are
quadrilaterals. Figure 4 also has a large triangular face. Wait, maybe the correct answer is 3. Wait, I think the answer is 3.</think><answer>To determine the shape after a cube loses one vertex (corner), we analyze the process: rem
oving a vertex creates a new triangular face where the corner was, modifying adjacent faces. Among the figures, the one with a large triangular face (matching the new face from cutting a corner) is identified. The correct figure is
3, so the final result is <|begin_of_box|>3<|end_of_box|>.', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[], reasoning_content=None), stop_reason=151348)], created=1757662384, model='
/home/mindone/yx/models/ZhipuAI/GLM-4.1V-9B-Thinking', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=1724, prompt_tokens=1217, total_tokens=2941, completion_tokens_deta
ils=None, prompt_tokens_details=None), prompt_logprobs=None, kv_transfer_params=None)

```

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Sponsors

vLLM is a community project. Our compute resources for development and testing are supported by the following organizations. Thank you for your support!

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
Cash Donations:

- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

Compute Resources:

- Alibaba Cloud
- AMD
- Anyscale
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Google Cloud
- Intel
- Lambda Lab
- Nebius
- Novita AI
- NVIDIA
- Replicate
- Roblox
- RunPod
- Trainy
- UC Berkeley
- UC San Diego

Slack Sponsor: Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
