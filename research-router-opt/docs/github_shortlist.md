# Agentic RL GitHub 选型记录

检索日期：2026-09-18。Star 为检索当日 GitHub 页面近似值，后续会变化。

| 项目 | 近似 Star | 能力定位 | 一天落地判断 |
|---|---:|---|---|
| [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) | 18.3k | 将真实 Agent harness、rollout、reward 与训练资源解耦 | 选为架构参考；原生 Windows local runner 不受支持 |
| [OpenPipe/ART](https://github.com/OpenPipe/ART) | 10.7k | 使用 GRPO 训练多步 Agent | 真实训练需要 GPU 或云后端，不适合作为一天首版 |
| [rllm-org/rllm](https://github.com/rllm-org/rllm) | 5.8k | 追踪 Agent 调用并形成 episode/trajectory/step，支持多种 RL 算法 | 后端和训练服务配置超出一天首版 |
| [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | 2.0k | 大规模异步 Agentic RL | 官方端到端链路需要 GPU，偏训练基础设施 |
| [AgentR1/Agent-R1](https://github.com/AgentR1/Agent-R1) | 1.7k | Step-level MDP 与多轮工具环境 | 设计值得借鉴，VERL 训练不适合当天完成 |
| [THUDM/slime](https://github.com/THUDM/slime) | 热门活跃 | 分布式 RL 后训练、异步 rollout、Coding/Multi-Agent RL | 面向大规模训练集群，不适合求职当天原型 |

## 选型结论

Agent Lightning 最符合 Agent/LLM 应用岗位的工程叙事：它强调已有 Agent、执行轨迹、奖励信号和
可优化资源之间的边界，而不是只展示训练算法。官方 APO 示例能够说明提示资源优化，但不等同于
PPO/GRPO 权重训练。

本机 WSL 服务不可用，而 Agent Lightning 官方 local runner 不支持原生 Windows。因此当天版本
没有伪装成 Agent Lightning 运行结果，也没有复制其代码；只借鉴 rollout–reward–resource
分层，独立实现 CPU/Windows 可验证的 LinUCB 工具路由实验。未来在 Linux/GPU 环境中接入
Agent Lightning 或 ART 时，应单独建立基线，不能沿用本项目的合成指标宣称权重训练效果。

## 简历归属规则

- 可以写“参考 Agent Lightning 分层思路，独立实现”；
- 不可以写“参与开发 Agent Lightning”；
- 不可以把 LinUCB 路由策略描述成 LLM 权重强化学习；
- 上游项目名称、链接和 License 必须保留；
- 个人贡献只包括本仓库的数据、环境、工具、奖励、策略、评测、API、测试和文档。
